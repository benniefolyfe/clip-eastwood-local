import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import certifi
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from slack_sdk import WebClient
from datetime import datetime
import re
from dotenv import load_dotenv
import sqlite3
from sqlite3 import Error
import json
import threading
import time
from datetime import datetime, timedelta
from slack_sdk.errors import SlackApiError
from google.api_core.exceptions import ResourceExhausted
from contextlib import contextmanager
import sys
import atexit
import html


os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Global variables
SEARCH_LIMIT = 3000 # Max messages to search
CHUNK_SIZE = 500 # Messages per Gemini call
DELAY_BETWEEN_CALLS = 10 # Seconds to wait between Gemini calls
MAX_CHUNKS = 6 # Hard cap to avoid runaway loops
SUPER_SEARCH_CHUNK_SIZE = 500
MAX_SOURCES = 10  # Adjust this number as needed
BACKFILL_INTERVAL = 1800 # 30 minutes
MESSAGE_LIMIT = 500 # Limits the number of messages to use in context in a private channel or DM
SEARCH_TRIGGERS = ["search", "find", "look", "show", "gather", "get"]
ENABLE_MENTION_INJECTION = True

# Initialize Slack clients
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)

# Cache the bot's user ID once
try:
    BOT_USER_ID = client.auth_test()['user_id']
except Exception as e:
    logging.error(f"Failed to fetch bot user ID: {e}")
    BOT_USER_ID = None

def safe_user_reference(user_id, fallback="partner"):
    """Return a Slack mention unless it's the bot itself."""
    if user_id == BOT_USER_ID:
        return fallback
    return f"<@{user_id}>"

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)

logger = logging.getLogger("slack_bot")

# Suppress excessive logging from external libraries
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

genai.configure(api_key=GEMINI_API_KEY)

def get_model():
    try:
        # Use the latest Gemini 2.0 Flash model
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logger.error("Failed to load gemini-2.0-flash, falling back to gemini-1.5-flash", exc_info=True)
        # Fallback to 1.5-flash only if still available, but this will be deprecated soon
        return genai.GenerativeModel('gemini-1.5-flash')

model = get_model()

DATABASE_FILE = "slack_messages.db"

@contextmanager
def db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize SQLite database with messages table"""
    with db_connection() as conn:
        c = conn.cursor()
        
        # Enable Write-Ahead Logging (WAL) and Optimize PRAGMAs
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
       
        # Main messages table
        c.execute('''CREATE TABLE IF NOT EXISTS messages
        (ts TEXT PRIMARY KEY,
        text TEXT,
        user_id TEXT,
        username TEXT,
        channel_id TEXT,
        channel_name TEXT,
        thread_ts TEXT,
        reactions TEXT,
        attachments TEXT,
        permalink TEXT,
        is_bot BOOLEAN,
        deleted BOOLEAN DEFAULT FALSE)''')
        
        # Last seen timestamps per channel
        c.execute('''CREATE TABLE IF NOT EXISTS last_seen
        (channel_id TEXT PRIMARY KEY,
        last_ts TEXT)''')
        
        # Whether messages have been responded to
        c.execute('''CREATE TABLE IF NOT EXISTS responded_messages
        (ts TEXT PRIMARY KEY)''')
        conn.commit()

init_db()

def get_slack_entity(entity_type, entity_id, cache, fetch_fn):
    if entity_id in cache:
        return cache[entity_id]
    try:
        if not entity_id:
            return f"Unknown {entity_type.title()}"
        entity_info = fetch_fn(entity_id)
        name = entity_info.get('name') or entity_info.get('real_name') or entity_id
        cache[entity_id] = name
        return name
    except Exception:
        return f"Unknown {entity_type.title()}"

def get_channel_and_thread_context(client, channel_id, limit=50, logger=None):
    """
    Fetches main channel messages and all threaded replies, merged and sorted chronologically.
    """
    # Fetch top-level messages
    try:
        response = client.conversations_history(channel=channel_id, limit=limit)
        top_level_msgs = response.get('messages', [])
    except SlackApiError as e:
        logger.error(f"Error fetching channel history: {e}")
        return []
    
    all_msgs = []
    for msg in top_level_msgs:
        # Add top-level message
        all_msgs.append(msg)
        
        # Fetch thread replies if available
        if msg.get('reply_count', 0) > 0 and msg.get('thread_ts'):
            try:
                thread_response = client.conversations_replies(
                    channel=channel_id,
                    ts=msg['thread_ts']
                )
                # Skip first message (it's the parent we already have)
                thread_replies = thread_response.get('messages', [])[1:]
                all_msgs.extend(thread_replies)
            except SlackApiError as e:
                logger.warning(f"Couldn't fetch thread {msg['thread_ts']}: {e}")
    
    # Sort chronologically
    try:
        all_msgs.sort(key=lambda m: float(m['ts']))
    except Exception:
        pass
    
    return all_msgs[-limit:]  # Return most recent messages

user_name_cache = {}
channel_name_cache = {}

def get_username(user_id):
    return get_slack_entity(
        "user", user_id, user_name_cache,
        lambda uid: client.users_info(user=uid)['user']
    )

def get_channel_name(channel_id):
    return get_slack_entity(
        "channel", channel_id, channel_name_cache,
        lambda cid: client.conversations_info(channel=cid)['channel']
    )

def build_mention_map():
    mention_map = {}
    users = client.users_list()["members"]
    for user in users:
        profile = user.get("profile", {})
        real_name = profile.get("real_name", "").strip()
        display_name = profile.get("display_name", "").strip()

        if not real_name:
            continue

        first_name = real_name.split()[0]

        # Add both full name and first name
        mention_map[real_name] = user["id"]
        mention_map[first_name] = user["id"]

        # Optional: add display name if different
        if display_name and display_name != first_name and display_name != real_name:
            mention_map[display_name] = user["id"]

    return mention_map

def assemble_message_data(msg, channel_info):
    """Build a dict with all message fields for DB upsert/insert."""
    ts = msg.get('ts')
    channel_id = channel_info['id']

    return {
        'ts': ts,
        'text': msg.get('text'),
        'user_id': msg.get('user'),
        'username': get_username(msg.get('user')) if msg.get('user') else 'Unknown User',
        'channel_id': channel_id,
        'channel_name': channel_info.get('name') or channel_info.get('user') or 'unknown',
        'thread_ts': msg.get('thread_ts'),
        'reactions': json.dumps(msg.get('reactions', [])),
        'attachments': json.dumps(msg.get('attachments', [])),
        'permalink': safe_get_permalink(client, channel_id, ts),
        'is_bot': int('bot_id' in msg)
    }

def filter_search_messages(messages):
    """
    Return only messages that are not deleted and not sent by bots.
    Handles Slack API format and common DB fields.
    """
    filtered = []
    for msg in messages:
        # Ignore deleted messages
        if msg.get('subtype') == 'message_deleted':
            continue
        if msg.get('deleted', False):
            continue
        if msg.get('is_deleted', False):
            continue
        # Ignore messages sent by bots
        if msg.get('subtype') == 'bot_message' or msg.get('bot_id') or (msg.get('message', {}).get('bot_id')):
            continue
        filtered.append(msg)
    return filtered

def save_or_update_message(msg, channel_info, action="upsert", source="event"):
    """
    Insert, update, or mark a message as deleted in the database.

    Args:
        msg (dict): Slack message dictionary (must include 'ts').
        channel_info (dict): Channel info dictionary (must include 'id').
        action (str): "upsert" for insert/update, "delete" for soft-delete.
        source (str): Event source, for logging/audit.
    """
    ts = msg.get('ts')
    channel_id = channel_info['id']
    
    if action == "delete":
        logger.info(f"[DEBUG] Attempting to mark message ts={ts} as deleted in channel={channel_id}.")
        with db_connection() as conn:
            c = conn.cursor()
            c.execute('UPDATE messages SET deleted = 1 WHERE ts = ? AND channel_id = ?', (ts, channel_id))
            conn.commit()
        logger.info(f"[DB] Marked message ts={ts} in channel {channel_id} as deleted.")
        return

    # Upsert logic
    text = msg.get('text')
    if text is None or text == "":
        logger.info(f"[DB] Skipping upsert for message ts={ts} in channel {channel_id} due to empty or None text.")
        return

    msg_data = assemble_message_data(msg, channel_info)
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM messages WHERE ts = ? AND channel_id = ?", (ts, channel_id))
        exists = c.fetchone() is not None
        if exists:
            c.execute('''
                UPDATE messages SET
                    text=:text,
                    user_id=:user_id,
                    username=:username,
                    channel_name=:channel_name,
                    thread_ts=:thread_ts,
                    reactions=:reactions,
                    attachments=:attachments,
                    permalink=:permalink,
                    is_bot=:is_bot,
                    deleted=0
                WHERE ts = :ts AND channel_id = :channel_id
            ''', msg_data)
        else:
            logger.info(f"[DEBUG] Attempting to upsert message ts={ts} in channel={channel_id}.")
            c.execute('''
                INSERT INTO messages (
                    ts, text, user_id, username, channel_id, channel_name,
                    thread_ts, reactions, attachments, permalink, is_bot, deleted
                ) VALUES (
                    :ts, :text, :user_id, :username, :channel_id, :channel_name,
                    :thread_ts, :reactions, :attachments, :permalink, :is_bot, 0
                )
            ''', msg_data)
        c.execute('INSERT OR REPLACE INTO last_seen (channel_id, last_ts) VALUES (?, ?)', (channel_id, ts))
        logger.info("[DB] Successfully added message to log.")
        conn.commit()

def safe_get_permalink(client, channel_id, ts):
    try:
        response = client.chat_getPermalink(channel=channel_id, message_ts=ts)
        if response.get('ok'):
            return response.get('permalink')
        else:
            logger.warning(f"Failed to get permalink for ts={ts} in channel={channel_id}: {response.get('error')}")
            return None
    except SlackApiError as e:
        if hasattr(e, "response") and e.response.get('error') == 'message_not_found':
            logger.warning(f"Message not found for ts={ts} in channel={channel_id}. It may have been deleted.")
            return None
        else:
            logger.error(f"Slack API error when getting permalink: {e}", exc_info=True)
            return None
    except Exception as e:
        if 'message_not_found' in str(e):
            logger.warning(f"Message not found for ts={ts} in channel={channel_id}. It may have been deleted.")
            return None
        else:
            logger.error(f"Slack API error when getting permalink: {e}", exc_info=True)
            return None

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def log_interaction(user_id, channel_id, message):
    user_name = get_username(user_id)
    channel_name = get_channel_name(channel_id)
    logger.info(f"[INTERACTION] User:{user_name} | Channel:{channel_name} | Message:'{message}'")

def log_mention(user_id, channel_id, message):
    user_name = get_username(user_id)
    channel_name = get_channel_name(channel_id)
    logger.info(f"[MENTION] From user {user_name} in channel {channel_name}: {message}")

def has_responded(ts):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT 1 FROM responded_messages WHERE ts = ?', (ts,))
        result = c.fetchone()
    return result is not None

def mark_as_responded(ts):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('INSERT OR IGNORE INTO responded_messages (ts) VALUES (?)', (ts,))
        conn.commit()

def inject_mentions(text):
    def safe_replace(pattern, replacement, text):
        def replacer(match):
            start, end = match.span()
            # Avoid replacing inside existing <@...> mentions
            if re.search(r'<@[^>]+>', text[max(0, start-3):min(end+3, len(text))]):
                return match.group(0)
            return replacement
        return re.sub(pattern, replacer, text)

    # Step 1: Replace user IDs
    for user_id in set(MENTION_MAP.values()):
        if user_id == BOT_USER_ID:
            continue  # Skip the bot itself
        pattern = r'(?<!<@)' + re.escape(user_id) + r'(?!>)'
        replacement = f'<@{user_id}>'
        text = safe_replace(pattern, replacement, text)

    # Step 2: Replace names
    for name in sorted(MENTION_MAP, key=len, reverse=True):
        user_id = MENTION_MAP[name]
        if user_id == BOT_USER_ID:
            continue  # Skip the bot itself
        pattern = r'\b' + re.escape(name) + r'\b'
        replacement = f'<@{user_id}>'
        text = safe_replace(pattern, replacement, text)

    return text

def is_search_request(message):
    return any(trigger in message.lower() for trigger in SEARCH_TRIGGERS) if message else False

def extract_search_terms_and_instruction(message, logger=None):

    extraction_prompt = f"""Extract from this message:
    1. Keywords to search for (comma-separated)
    2. Instruction for presenting results (after ";")
    
    Example response for "Find docs about AI and summarize key points"
    
    "AI, documentation; summarize key points"
    
    Message: "{message}
    
    Response:"""

    try:
        response = generate_response(extraction_prompt, safe=True).strip()
        if ";" in response:
            terms_part, instruction = response.split(";", 1)
            return [t.strip() for t in terms_part.split(",")], instruction.strip()
        else:
            return [response], "Summarize the findings"
    except Exception as e:
        if logger:
            logger.error(f"Search term extraction failed: {e}")
        return [message], "Summarize the findings"

def is_message_too_old(ts, hours=24):
    try:
        msg_time = datetime.fromtimestamp(float(ts))
        return datetime.now() - msg_time > timedelta(hours=hours)
    except Exception:
        return True  # If parsing fails, treat as too old

def search_messages_local(terms, channel_id=None):
    """Search local database for messages containing any of the terms (OR logic)"""
    try:
        with db_connection() as conn:
            c = conn.cursor()
            query = '''
            SELECT * FROM messages
            WHERE deleted = 0
            AND ({})
            '''.format(' OR '.join(['text LIKE ?'] * len(terms)))
            params = [f'%{term}%' for term in terms]
            if channel_id:
                query += ' AND channel_id = ?'
                params.append(channel_id)
            c.execute(query, params)
            rows = c.fetchall()
            results = [dict(row) for row in rows]
        return results
    except Error as e:
        logger.error(f"[DB] Search error: {e}", exc_info=True)
        return []

def get_last_seen(channel_id):
    """Get last processed timestamp for a channel"""
    try:
        with db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT last_ts FROM last_seen WHERE channel_id = ?', (channel_id,))
            result = c.fetchone()
            last_ts = result['last_ts'] if result else '0'
        return last_ts
    except Error as e:
        logger.error(f"[DB] Error getting last seen: {e}", exc_info=True)
        return '0'

def message_exists(ts, channel_id):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM messages WHERE ts=? AND channel_id=?", (ts, channel_id))
        return c.fetchone() is not None

def periodic_backfill(interval=BACKFILL_INTERVAL):
    """Periodic backfill scheduler with auto-vacuum and null entry cleanup"""
    while True:
        try:
            backfill_and_process_mentions()
            
            # Database maintenance: VACUUM and remove null/empty entries
            with db_connection() as conn:
                conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                conn.execute("PRAGMA incremental_vacuum")
                logger.info("[BACKFILL] Automatic VACUUM complete")
                conn.execute("DELETE FROM messages WHERE text IS NULL OR text = ''")
                conn.commit()
                logger.info("[BACKFILL] Removed NULL/empty text entries")

        except Exception as e:
            logger.error(f"[BACKFILL] Periodic backfill error: {e}", exc_info=True)
        
        time.sleep(interval)

def backfill_and_process_mentions():
    """Backfill missed messages and process mentions (with pagination), skipping already logged messages."""
    try:
        logger.info("[BACKFILL] Starting backfill process")
        channels = client.conversations_list(types="public_channel")["channels"]

        for channel in channels:
            channel_id = channel['id']
            if not channel.get('is_member', False):
                continue

            last_ts = get_last_seen(channel_id)
            logger.debug(f"[BACKFILL] Backfilling {channel['name']} from {last_ts}")

            cursor = None
            while True:
                response = client.conversations_history(
                    channel=channel_id,
                    oldest=last_ts,
                    limit=1000,
                    cursor=cursor
                )

                for msg in response['messages']:
                    ts = msg.get('ts')
                    if not ts:
                        continue

                    # Skip bot join/system messages
                    if "has joined the channel" in msg.get('text', '').lower():
                        continue

                    # Skip if message is too old
                    if is_message_too_old(ts):
                        continue

                    # Skip if already responded
                    if has_responded(ts):
                        continue

                    # Skip if already logged in DB
                    with db_connection() as conn:
                        c = conn.cursor()
                        c.execute("SELECT 1 FROM messages WHERE ts=? AND channel_id=?", (ts, channel_id))
                        if c.fetchone():
                            logger.info(f"[BACKFILL] Skipping already-logged message ts={ts} in channel={channel_id}")
                            continue

                    # Store message
                    store_message_with_check(msg, channel)

                    # Check for mentions
                    if f"<@{BOT_USER_ID}>" in msg.get('text', ''):
                        log_mention(msg.get('user'), channel_id, msg.get('text'))
                        handle_app_mention({
                            'user': msg.get('user'),
                            'channel': channel_id,
                            'text': msg.get('text'),
                            'ts': ts,
                            'thread_ts': msg.get('thread_ts')
                        }, lambda text, **kwargs: client.chat_postMessage(
                            channel=channel_id,
                            text=text,
                            thread_ts=msg.get('thread_ts') if msg.get('thread_ts') else None
                        ), client, logger
                    )
                    mark_as_responded(ts)

                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break

        logger.info("[BACKFILL] Backfill complete")

    except Exception as e:
        logger.error(f"[BACKFILL] Critical error: {e}", exc_info=True)

def get_reply_thread_ts(event):
    """
    Returns the correct thread_ts for replying:
    - If the message is in a thread (thread_ts present and not equal to ts), return thread_ts.
    - Otherwise, return None (reply in channel).
    """
    ts = event.get("ts")
    thread_ts = event.get("thread_ts")
    if thread_ts and thread_ts != ts:
        return thread_ts
    return None

@app.event("message")
def handle_all_messages(event, say, logger):
    # Ignore bot messages and channel join messages
    if event.get("subtype") in ("bot_message", "channel_join"):
        return

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text", "")

    if "has joined the channel" in text.lower():
        return

    # Fetch channel info to determine type
    try:
        channel_info = client.conversations_info(channel=channel_id)["channel"]
        is_dm = channel_info.get("is_im", False)
        is_private = channel_info.get("is_private", False)
    except Exception as e:
        logger.error(f"Error fetching channel info: {e}")
        is_dm = False
        is_private = False

    # Store message in DB only for public channels
    if not (is_dm or is_private):
        try:
            store_message_with_check(event, channel_info)
        except Exception as e:
            logger.error(f"[DB] Error storing message: {e}", exc_info=True)

    # Handle message deletions
    if event.get("subtype") == "message_deleted":
        deleted_ts = event.get("deleted_ts")
        msg = {'ts': deleted_ts}
        save_or_update_message(msg, channel_info, action="delete")
        return

    # Handle message edits
    if event.get("subtype") == "message_changed":
        edited_msg = event["message"]
        store_message_with_check(edited_msg, channel_info)
        return

    # Handle DMs: respond to every message using Slack API context
    if is_dm:
        handle_dm_message(event, say, client, logger)
        return

    # Private channels: do NOT auto-respond here.

def handle_dm_message(event, say, client, logger):
    """
    Responds to every message in a DM using context from the DM via Slack API.
    """
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts")
    message_ts = event.get("ts")
    text = event.get("text", "")

    try:
        # Fetch recent DM messages for context
        resp = client.conversations_history(channel=channel_id, limit=MESSAGE_LIMIT)
        messages = list(reversed(resp.get("messages", [])))
    except Exception as e:
        logger.error(f"[DM] Error fetching message context: {e}")
        say("Sorry, I couldn't fetch the conversation context.")
        return

    # Build context for prompt
    context_messages = []
    for msg in messages:
        if 'text' in msg and msg.get('user'):
            try:
                sender = get_username(msg['user'])
                timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
            except Exception as e:
                logger.warning(f"[CONTEXT] Skipped message due to user info error: {e}")
        elif 'text' in msg:  # Bot or system message
            try:
                sender = msg.get('username', 'Clip Eastwood')
                timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
            except Exception as e:
                logger.warning(f"[CONTEXT] Skipped message due to missing user: {e}")

    primary_message = text
    context = {
        "primary_message": primary_message,
        "previous_messages": "\n".join(context_messages[-MESSAGE_LIMIT:])
    }
    prompt = build_context_prompt(context)
    response_text = generate_response(prompt)
    formatted = format_for_slack(response_text or "")

    # Reply in thread if possible
    reply_thread_ts = get_reply_thread_ts(event)
    if reply_thread_ts:
        say(formatted, thread_ts=reply_thread_ts, mrkdwn=True)
    else:
        say(formatted, mrkdwn=True)

def handle_private_message(event, say, client, logger):
    """
    Responds to @mentions in private channels using Slack API for context.
    """
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts")
    message_ts = event.get("ts")
    text = event.get("text", "")

    try:
        if thread_ts:
            # Fetch thread context
            messages = []
            cursor = None
            while True:
                resp = client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    cursor=cursor,
                    limit=MESSAGE_LIMIT
                )
                messages.extend(resp.get("messages", []))
                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
        else:
            # Fetch recent messages from the private channel, including thread replies, for context
            messages = get_channel_and_thread_context(client, channel_id, limit=MESSAGE_LIMIT, logger=logger)
    except Exception as e:
        logger.error(f"[PRIVATE CHANNEL] Error fetching message context: {e}")
        say("Sorry, I couldn't fetch the conversation context.")
        return

    # Build context for prompt
    context_messages = []
    for msg in messages:
        if 'text' in msg and msg.get('user'):
            try:
                sender = get_username(msg['user'])
                timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
            except Exception as e:
                logger.warning(f"[CONTEXT] Skipped message due to user info error: {e}")
        elif 'text' in msg:  # Bot or system message
            try:
                sender = msg.get('username', 'Clip Eastwood')
                timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
            except Exception as e:
                logger.warning(f"[CONTEXT] Skipped message due to missing user: {e}")

    primary_message = text
    context = {
        "primary_message": primary_message,
        "previous_messages": "\n".join(context_messages[-MESSAGE_LIMIT:])
    }
    prompt = build_context_prompt(context)
    response_text = generate_response(prompt)
    formatted = format_for_slack(response_text or "")

    # Reply in thread if possible
    reply_thread_ts = get_reply_thread_ts(event)
    if reply_thread_ts:
        say(formatted, thread_ts=reply_thread_ts, mrkdwn=True)
    else:
        say(formatted, mrkdwn=True)

def store_message_with_check(msg, channel_info, source="event"):
    """Skips storage for DMs and private channels"""
    channel_id = channel_info['id']
    is_dm = channel_info.get("is_im", False)
    is_private = channel_info.get("is_private", False)
    
    # Skip DMs and private channels
    if is_dm or is_private:
        logger.info(f"[DB] Skipping message in {'DM' if is_dm else 'private channel'} {channel_id}")
        return

    # Existing storage logic for public channels
    text = msg.get('text')
    ts = msg.get('ts')
    
    if text is None and source == "deleted_event":
        save_or_update_message(msg, channel_info, action="delete", source=source)
        return
        
    logger.info(f"[DB] Upserting message ts={ts} in channel={channel_id} (source={source}).")
    save_or_update_message(msg, channel_info, action="upsert", source=source)

@app.event("app_mention")
def handle_app_mention(event, say, client, logger):
    ts = event.get("ts")
    user = event.get("user")
    subtype = event.get("subtype")
    bot_id = event.get("bot_id")
    channel_id = event.get("channel")

    # 🛑 Prevent the bot from responding to its own messages
    if user == BOT_USER_ID:
        logger.info(f"[MENTION] Skipped responding to self-mention at ts={ts}")
        return

    # Ignore bot messages and already processed messages
    if subtype == "bot_message" or bot_id is not None:
        return

    if has_responded(ts):
        logger.info(f"Skipping already processed mention: {ts}")
        return

    mark_as_responded(ts)

    if event.get("subtype") == "channel_join":
        logger.info("[MENTION] Ignored app_mention event with subtype channel_join.")
        return

    # Fetch channel info to determine if private
    try:
        channel_info = client.conversations_info(channel=channel_id)["channel"]
        is_private = channel_info.get("is_private", False)
    except Exception as e:
        logger.error(f"[MENTION] Error fetching channel info: {e}")
        is_private = False

    # If in a private channel, use the private channel handler
    if is_private:
        handle_private_message(event, say, client, logger)
        return

    original_text = event.get('text', '')
    channel = event.get('channel')
    user = event.get('user')

    log_mention(user, channel, original_text)

    try:
        if BOT_USER_ID:
            cleaned_text = original_text.replace(f"<@{BOT_USER_ID}>", '').strip()
            event['text'] = cleaned_text
            logger.debug(f"[MENTION] Cleaned text passed to interaction handler: '{cleaned_text}'")
        else:
            logger.warning("[MENTION] BOT_USER_ID is not set — skipping mention cleanup.")

        # Pass client and logger to handle_bot_interaction
        handle_bot_interaction(event, say, client, logger)
    except Exception as e:
        logger.error(f"[MENTION] Error processing mention: {e}", exc_info=True)

def format_source_for_thread(msg, idx=None):
    """
    Format a source message for posting as a threaded reply.
    """
    try:
        ts_float = float(msg['ts'])
        dt = datetime.fromtimestamp(ts_float)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        timestamp_str = msg['ts']

    author = get_username(msg['user_id']) if msg.get('user_id') else "Unknown"
    channel = msg.get('channel_name', 'unknown')
    text_preview = (msg.get('text', '') or '')[:200].replace('\n', ' ')
    permalink = msg.get('permalink')
    deleted = msg.get('deleted', 0)
    deleted_note = " _(deleted)_" if deleted else ""

    link = f"<{permalink}|link>" if permalink else "(unavailable)"
    prefix = f"{idx}. " if idx else ""

    return (f"{prefix}[{timestamp_str}, @{author}, #{channel}]{deleted_note}: "
            f"\"{text_preview}\" {link}")

def handle_search_request(
    keywords, instruction, user_id, channel_id, say, client, logger, source="command", super_search=False
):
    """
    Handles both standard and super search requests.
    - If super_search: searches ALL messages, no keyword filtering, no chunk limit.
    - Otherwise: standard search with keywords and chunk limit.
    """
    logger.info(
        f"[SEARCH][{source.upper()}] Handling request - "
        f"Keywords: {keywords} | Instruction: {instruction} | SuperSearch: {super_search}"
    )

    if super_search:
        # Fetch ALL messages, no keyword filtering, no limit
        raw_results = search_messages_local_flat(limit=None)
        chunk_size = SUPER_SEARCH_CHUNK_SIZE
        max_chunks = float('inf')  # No artificial chunk limit
    else:
        raw_results = search_messages_across_channels(keywords, logger=logger, search_limit=SEARCH_LIMIT)
        chunk_size = CHUNK_SIZE
        max_chunks = MAX_CHUNKS

    results = filter_search_messages(raw_results)

    if not results:
        say(f"No messages found for: {', '.join(keywords) if keywords else 'all messages'}", channel=channel_id)
        return

    summaries = []
    sources = results

    try:
        for i, chunk in enumerate(chunk_list(results, chunk_size)):
            if i >= max_chunks:
                summaries.append("Too many results. Please narrow your search.")
                break

            logger.info(
                f"[SEARCH][CHUNK] Summarizing chunk {i+1}/"
                f"{'∞' if super_search else max_chunks} with {len(chunk)} messages."
            )
            context_snippets = [format_message_for_context(r) for r in chunk]
            prompt = (
                f"You are Clip Eastwood, a Slack-based creative assistant. "
                f"Below are Slack messages{' from our entire history' if super_search else f' about {', '.join(keywords)}'}. "
                f"{instruction}\n\n"
                f"{'-'*20}\n"
                f"{chr(10).join(context_snippets)}"
            )
            summary = generate_response(prompt, safe=True)
            summaries.append(summary)
            time.sleep(DELAY_BETWEEN_CALLS)
    except ResourceExhausted:
        say(
            "Sorry, the AI service is temporarily unavailable due to quota limits. Please try again later.",
            channel=channel_id,
        )
        return

    if len(summaries) > 1:
        logger.info(f"[SEARCH][FINAL] Summarizing {len(summaries)} chunk summaries into final result.")
        final_prompt = (
            f"Summarize the following summaries into a single cohesive answer:\n\n"
            f"{'-'*20}\n"
            f"{chr(10).join(summaries)}"
        )
        try:
            final_summary = generate_response(final_prompt, safe=True)
        except ResourceExhausted:
            say(
                "Sorry, the AI service is temporarily unavailable due to quota limits. Please try again later.",
                channel=channel_id,
            )
            return
    else:
        final_summary = summaries[0] if summaries else "No summary available."

    sources_to_show = sources[:MAX_SOURCES]
    sources_note = ""
    if len(sources) > MAX_SOURCES:
        sources_note = f"_{len(sources)} sources found._"
    elif sources:
        sources_note = f"_{len(sources)} sources found._"
    else:
        sources_note = "_No relevant sources found._"

    # Post summary with sources note
    context_header = (
        f"*Requestor:* <@{user_id}>\n"
        f"*Search Type:* {'SUPER SEARCH' if super_search else 'Standard'}\n"
        f"*Keywords:* {', '.join(keywords) if keywords else '(all messages)'}\n"
        f"*Context:* {instruction}\n\n"
        f"{sources_note}\n\n"
    )

    summary_post = client.chat_postMessage(
        channel=channel_id,
        text=context_header + "\n" + format_for_slack(final_summary or ""),
    )
    parent_ts = summary_post["ts"]

@app.command("/ai-search")
def handle_search_command(ack, respond, command, say, logger, client):
    ack()
    text = command.get("text", "").strip()
    user_id = command.get("user_id")
    channel_id = command.get("channel_id")

    if ";" in text:
        terms_part, instruction = text.split(";", 1)
        instruction = instruction.strip()
    else:
        terms_part = text
        instruction = "Summarize the findings"

    keywords = [t.strip() for t in terms_part.split(",") if t.strip()]

    respond(
        f"Searching for {', '.join(keywords)} across channels and fulfilling your request to '{instruction}'. "
        f"This may take a few minutes..."
    )

    handle_search_request(
        keywords=keywords,
        instruction=instruction,
        user_id=user_id,
        channel_id=channel_id,
        say=say,
        client=client,
        logger=logger,
        source="command",
        super_search=False,
    )

@app.command("/ai-super-search")
def handle_super_search_command(ack, respond, command, say, logger, client):
    ack()
    text = command.get("text", "").strip()
    user_id = command.get("user_id")
    channel_id = command.get("channel_id")

    # For super search, treat the full text as instruction (no keywords)
    instruction = text if text else "Summarize the findings"

    respond(
        f"Running SUPER SEARCH across ALL messages. Instruction: '{instruction}'. This may take up to 10 minutes..."
    )

    handle_search_request(
        keywords=[],  # No keywords for super search
        instruction=instruction,
        user_id=user_id,
        channel_id=channel_id,
        say=say,
        client=client,
        logger=logger,
        source="command",
        super_search=True,
    )

def search_messages_local_flat(limit=None, channel_id=None):
    """Fetch all (or recent) messages from local DB, with optional limit."""
    try:
        with db_connection() as conn:
            c = conn.cursor()
            query = "SELECT * FROM messages WHERE deleted = 0"
            params = []
            if channel_id:
                query += " AND channel_id = ?"
                params.append(channel_id)
            query += " ORDER BY ts DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            c.execute(query, params)
            rows = c.fetchall()
            results = [dict(row) for row in rows]
        return results
    except Error as e:
        logger.error(f"[DB] Search error: {e}", exc_info=True)
        return []

def search_messages_across_channels(search_terms, logger=None, search_limit=None):
    """
    Search for messages containing any of the search terms (OR logic).
    Returns a list of relevant messages with full metadata.
    """
    try:
        results = []
        with db_connection() as conn:
            c = conn.cursor()
            query = ("SELECT * FROM messages "
                     "WHERE deleted = 0 AND (" +
                     " OR ".join(["text LIKE ?" for _ in search_terms]) +
                     ") ORDER BY ts DESC")
            params = [f'%{term}%' for term in search_terms]
            if search_limit:
                query += ' LIMIT ?'
                params.append(search_limit)
            c.execute(query, params)
            rows = c.fetchall()
            results = [dict(row) for row in rows]
        # Remove duplicates (by ts + channel_id)
        seen = set()
        unique_results = []
        for r in results:
            key = (r["channel_id"], r["ts"])
            if key not in seen:
                unique_results.append(r)
                seen.add(key)
        return unique_results
    except Exception as e:
        if logger:
            logger.error(f"[SEARCH] Error searching messages: {e}", exc_info=True)
        return []

def handle_bot_interaction(event, say, client, logger):
    """Handles @mentions and DMs"""
    start_time = time.time()
    user_id = event.get('user')
    channel_id = event.get('channel')
    message_text = event.get('text', '')
    primary_message = event.get('text')
    thread_ts = event.get('thread_ts')

    # Ignore bot messages
    if event.get('bot_id') or event.get('subtype') == 'bot_message':
        return

    # Route search requests
    if is_search_request(message_text):
        keywords, instruction = extract_search_terms_and_instruction(message_text, logger)

        # Send ephemeral "searching" message to the user
        try:
            client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=(
                    f"Searching for {', '.join(keywords)} across channels and fulfilling your request to '{instruction}'. This may take a few minutes..."
                ),
            )
        except Exception as e:
            logger.error(f"Failed to send ephemeral search message: {e}")

        handle_search_request(
            keywords=keywords,
            instruction=instruction,
            user_id=user_id,
            channel_id=channel_id,
            say=say,
            client=client,
            logger=logger,
            source="mention"
        )
        return

    # If not a search request, proceed with normal contextual response
    log_interaction(user_id, channel_id, primary_message)
    context_messages = []
    # In handle_bot_interaction:
    try:
        if thread_ts:
            # If in a thread, fetch just the thread context
            messages = get_thread_and_context_messages(channel_id, thread_ts, limit=CHUNK_SIZE)
        else:
            # If not in a thread (top-level mention), fetch top-level + thread replies for full context
            messages = get_channel_and_thread_context(client, channel_id, limit=CHUNK_SIZE, logger=logger)
        logger.debug(f"[CONTEXT] Fetched {len(messages)} messages for context (thread_ts={thread_ts})")
        context_messages = []
        for msg in messages:
            if 'text' in msg and msg.get('user'):
                try:
                    sender = get_username(msg['user'])
                    timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                    context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
                except Exception as e:
                    logger.warning(f"[CONTEXT] Skipped message due to user info error: {e}")
            elif 'text' in msg:  # Bot or system message
                try:
                    sender = msg.get('username', 'Clip Eastwood')
                    timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                    context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
                except Exception as e:
                    logger.warning(f"[CONTEXT] Skipped message due to missing user: {e}")
        
        context = {
            "primary_message": primary_message,
            "previous_messages": "\n".join(context_messages[-CHUNK_SIZE:])
        }
        prompt = build_context_prompt(context)
        response_text = generate_response(prompt)
        logger.info(f"[RESPONSE] AI generated response for user {user_id}")
        formatted = format_for_slack(response_text or "")
        say(formatted, thread_ts=thread_ts if thread_ts else None, mrkdwn=True)
    except Exception as e:
        logger.error(f"[CONTEXT] Error fetching context or generating response: {e}", exc_info=True)
        say(f"Hey there, <@{user_id}>! I ran into a problem gathering context.")
    duration = time.time() - start_time
    logger.info(f"[PERF] Interaction completed in {duration:.2f}s for user {user_id}")


def get_thread_and_context_messages(channel_id, thread_ts=None, limit=CHUNK_SIZE):
    """
    Fetches the most recent messages for context, prioritizing thread if present,
    and always includes both user and bot messages.
    """
    messages = []
    cursor = None
    try:
        if thread_ts:
            # Fetch the full thread (including bot and user messages)
            while True:
                response = client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    cursor=cursor,
                    limit=min(limit, 200)  # Slack's max per page
                )
                messages.extend(response.get('messages', []))
                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor or len(messages) >= limit:
                    break
        else:
            # Fetch the last N messages from the channel (not just user)
            response = client.conversations_history(
                channel=channel_id,
                limit=limit
            )
            messages = list(reversed(response.get('messages', [])))
        # Sort by timestamp ascending (oldest first)
        messages = sorted(messages, key=lambda m: float(m['ts']))
        return messages[-limit:]  # Always return up to the last N messages
    except SlackApiError as e:
        logger.error(f"Error fetching messages: {e}", exc_info=True)
        return []

def build_context_prompt(context):
    """Prompt for context-rich user interactions (direct mentions, DMs)."""
    return f"""You are Clip Eastwood, a Slack-based creative assistant with a gruff, witty, and cowboy flair. 
    Your top priority is to follow user instructions quickly and accurately, using clarity and insight. 
    Be helpful, insightful, thorough in your responses, and capable of creative tasks and effective business communications 
    (lists, memos, letters, summaries, newsletters, blogs, articles, and multi-paragraph write-ups and summaries). 
    Respond in markdown, and maintain your signature style only in communicating with users, not in content output. 
    
    Instructions: 
    - Use the recent messages to understand any names, tasks, or references.
    - Do NOT repeat or summarize the context block—just use it to inform your reply.
    - Avoid repeating questions the user already answered.
    - Be warm and human, but don't delay the task.
    - Use wit and charm only to enhance clarity or delight, not to stall.
    
    User request:
    
    \"\"\"{context['primary_message']}\"\"\"
    
    Channel context:
    
    \"\"\"{context['previous_messages']}\"\"\"
    
    Respond clearly, confidently, and in a format that fits the request (lists, memos, letters, summaries, 
    newsletters, blogs, articles, and multi-paragraph write-ups and summaries). Prioritize completion."""

def generate_response(prompt, safe=True, max_retries=2):
    """
    Calls Gemini (or other LLM) with a prompt and returns a formatted Slack response.
    """
    for attempt in range(max_retries if safe else 1):
        try:
            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            text = getattr(response, "text", None)
            if not text:
                return "I couldn't generate a response."
            return format_for_slack(text)
        except ResourceExhausted as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) * 15
                logger.warning(f"API quota exceeded, retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Gemini API quota exceeded: {e}", exc_info=True)
                return "Sorry, the AI service is temporarily unavailable due to quota limits. Please try again later."
        except Exception as e:
            logger.error(f"AI generation failed: {e}", exc_info=True)
            return "Sorry, I couldn't generate a response due to an internal error."

def format_for_slack(text, do_inject_mentions=True):
    """End-to-end formatting pipeline for Gemini -> Slack, with - for bullets instead of *."""
    # Cleanup
    text = re.sub(r' {2,}', ' ', text)
    text = html.unescape(text)  # Convert HTML entities
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)  # Remove any remaining hex artifacts
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip('"\'')

    # Markdown conversion
    text = re.sub(r'^#+\s(.+)$', r'*\1*', text, flags=re.MULTILINE)  # Headers
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)  # Bold
    text = re.sub(r'`([^`]+)`', r'`\1`', text)  # Inline code

    # Link formatting
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<\2|\1>', text)

    # Bullet conversion: Replace lines starting with "* " (not already inside code blocks) with "- "
    text = re.sub(r'^( *)(\* )', r'\1- ', text, flags=re.MULTILINE)

    # Mention injection
    if do_inject_mentions and ENABLE_MENTION_INJECTION:
        text = inject_mentions(text)

    return text.strip()

MENTION_MAP = build_mention_map()

def format_message_for_context(msg):
    """
    Format a message for use in context or as a source reference.
    Includes timestamp, author, channel, and permalink.
    """
    # Format timestamp
    try:
        ts_float = float(msg['ts'])
        dt = datetime.fromtimestamp(ts_float)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        timestamp_str = msg['ts']

    author = get_username(msg['user_id']) if msg.get('user_id') else "Unknown"
    channel = msg.get('channel_name', 'unknown')
    text = msg.get('text', '')

    # Add reactions
    reactions = ""
    try:
        rx = json.loads(msg['reactions']) if msg['reactions'] else []
        if rx:
            reactions = " [Reactions: " + " ".join([f":{r['name']}|{r['count']}" for r in rx]) + "]"
    except Exception:
        pass

    # Add attachments
    attachments = ""
    try:
        att = json.loads(msg['attachments']) if msg['attachments'] else []
        file_links = []
        for a in att:
            if 'title' in a and 'title_link' in a:
                file_links.append(f"{a['title']}: {a['title_link']}")
            elif 'name' in a and 'url_private' in a:
                file_links.append(f"{a['name']}: {a['url_private']}")
        if file_links:
            attachments = f" [Attachments: {'; '.join(file_links)}]"
    except Exception:
        pass

    # Permalink (if available and not deleted)
    permalink = msg.get('permalink')
    deleted = msg.get('deleted', 0)
    deleted_note = " _(deleted)_" if deleted else ""

    # Compose
    return f"[{timestamp_str}, @{author}, #{channel}]{deleted_note}: {text}{reactions}{attachments}"

def join_all_channels():
    channels = client.conversations_list(types="public_channel")["channels"]
    logger.info("[STARTUP] Joining all public channels.")
    for channel in channels:
        channel_id = channel["id"]
        try:
            client.conversations_join(channel=channel_id)
        except Exception as e:
            # Ignore if already in channel or can't join
            pass
    logger.info("[STARTUP] Channels successfully joined.")    

def remove_null_text_entries():
    """Remove any messages with NULL text from the database."""
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE text IS NULL;")
        conn.commit()
    logger.info("[BACKFILL] Removed all messages with NULL text from the database.")

# Global stop signal for background threads
stop_event = threading.Event()

def main():
    # Register cleanup FIRST
    atexit.register(lambda: print("[EXIT] Interrupted by user."))
    
    # Start Bolt app in non-daemon thread
    bolt_thread = threading.Thread(
        target=lambda: SocketModeHandler(app, SLACK_APP_TOKEN).start(),
        daemon=False
    )
    bolt_thread.start()
    
    # Start background tasks as daemon threads
    threading.Thread(target=join_all_channels, daemon=True).start()
    threading.Thread(target=periodic_backfill, daemon=True).start()
    
    try:
        bolt_thread.join()  # Block until Bolt exits
    except KeyboardInterrupt:
        sys.exit(0)  # atexit handles cleanup

if __name__ == "__main__":
    main()