import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import certifi
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from slack_sdk import WebClient
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import sqlite3
from sqlite3 import Error
import json
import threading
import time
from slack_sdk.errors import SlackApiError
from google.api_core.exceptions import ResourceExhausted
from contextlib import contextmanager
import sys
import atexit
import html
import asana
import math

os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
ASANA_TOKEN = os.getenv("ASANA_TOKEN")

# Asana configuration
configuration = asana.Configuration()
configuration.access_token = ASANA_TOKEN
asana_api_client = asana.ApiClient(configuration)

# Create API instances for each resource you want to use
tasks_api = asana.TasksApi(asana_api_client)
projects_api = asana.ProjectsApi(asana_api_client)

# Global variables
CHUNK_SIZE = 500 # Messages per Gemini call
DELAY_BETWEEN_CALLS = 0 # Seconds to wait between Gemini calls
MAX_SOURCES = 10  # Adjust this number as needed
BACKFILL_INTERVAL = 1800 # 30 minutes
MESSAGE_LIMIT = 500 # Limits the number of messages to use in context in a private channel or DM
ENABLE_MENTION_INJECTION = True
SEARCH_TRIGGERS = ["search", "find", "look", "show", "gather", "get"]
GENERIC_STOPWORDS = {"project", "projects", "task", "tasks", "info", "information", "details", "summary", "summaries", "thing", "things", "item", "items", "topic", "topics"}
BOT_PERSONALITY_PROMPT = (
    "You are Clip Eastwood, a Slack-based creative assistant with a gruff, witty, cowboy flair. "
    "Greet and sign off in your signature style, but keep all summaries, lists, and business content professional and neutral unless the user requests otherwise. "
    "Use recent messages only as context—do not repeat or summarize them directly. "
    "Be helpful, concise, and insightful. Respond in markdown. "
    "If unsure, err on the side of clarity and professionalism."
)

# Initialize Slack clients
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)

def human_time(iso_str):
    try:
        return datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%B %d, %Y at %I:%M %p")
    except Exception:
        return iso_str or "N/A"

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

def get_default_workspace_id():
    workspaces_api = asana.WorkspacesApi(asana_api_client)
    workspaces = list(workspaces_api.get_workspaces({}))
    if not workspaces:
        raise Exception("No Asana workspaces available for your token.")
    return workspaces[0]['gid']

def init_db():
    """Initialize SQLite database with messages table and related tables."""
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

        # Asana tasks
        c.execute('''CREATE TABLE IF NOT EXISTS asana_tasks (
        gid TEXT PRIMARY KEY,
        name TEXT,
        notes TEXT,
        assignee TEXT,
        completed BOOLEAN,
        due_on TEXT,
        project_id TEXT,
        modified_at TEXT,
        custom_fields_json TEXT)''')

        # Asana projects
        c.execute('''CREATE TABLE IF NOT EXISTS asana_projects (
        gid TEXT PRIMARY KEY,
        name TEXT,
        notes TEXT,
        created_at TEXT,
        modified_at TEXT,
        custom_fields_json TEXT)''')

        # Asana custom field definitions
        c.execute('''CREATE TABLE IF NOT EXISTS asana_custom_fields (
        gid TEXT PRIMARY KEY,
        name TEXT,
        type TEXT,
        enum_options_json TEXT)''')

        # Commit after all tables are created
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

def get_channel_and_thread_context(client, channel_id, limit=500, logger=None):
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

def format_custom_fields(custom_fields_json):
    """Format custom fields JSON into a readable string, using field definitions."""
    try:
        fields = json.loads(custom_fields_json) if custom_fields_json else []
        if not fields:
            return ""
        # Load custom field definitions from DB
        with db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT gid, name, type, enum_options_json FROM asana_custom_fields")
            cf_defs = {row["gid"]: dict(row) for row in c.fetchall()}
        lines = []
        for field in fields:
            gid = field.get("gid")
            defn = cf_defs.get(gid, {})
            name = defn.get("name", f"Field {gid}")
            value = field.get("display_value") or field.get("text_value") or field.get("enum_value", {}).get("name") or field.get("number_value") or field.get("date_value") or ""
            if value is None:
                value = ""
            lines.append(f"    - {name}: {value}")
        return "\n".join(lines)
    except Exception as e:
        return ""

def get_asana_context_blocks(keywords=None):
    with db_connection() as conn:
        c = conn.cursor()
        tasks = [dict(row) for row in c.execute("SELECT * FROM asana_tasks").fetchall()]
        projects = [dict(row) for row in c.execute("SELECT * FROM asana_projects").fetchall()]

    from collections import defaultdict
    project_tasks = defaultdict(list)
    for task in tasks:
        project_id = task.get("project_id")
        task_str = f"- {task.get('name')} [Due: {human_time(task.get('due_on'))} | Completed: {bool(task.get('completed'))} | Last Updated: {human_time(task.get('modified_at'))}]"
        notes = task.get("notes", "")
        clean_notes = re.sub(r'https?://\S+', '[link removed]', notes)
        clean_notes = re.sub(r'\s+', ' ', clean_notes).strip()
        if clean_notes:
            task_str += f"\n  Notes: {clean_notes[:1000]}..."
        # Add custom fields for the task
        custom_fields_str = format_custom_fields(task.get("custom_fields_json"))
        if custom_fields_str:
            task_str += f"\n  Custom Fields:\n{custom_fields_str}"
        project_tasks[project_id].append(task_str)

    # If keywords provided, filter projects
    if keywords:
        keywords_lower = [k.lower() for k in keywords]
        projects = [
            p for p in projects
            if any(k in (p.get("name", "").lower() + " " + p.get("notes", "").lower()) for k in keywords_lower)
        ]

    formatted_blocks = []
    for project in projects:
        pid = project.get("gid")
        header = f"*Project: {project.get('name')}*\nLast Updated: {human_time(project.get('modified_at'))}"
        notes = f"Notes: {project.get('notes')[:1000]}." if project.get("notes") else ""
        # Add custom fields for the project
        custom_fields_str = format_custom_fields(project.get("custom_fields_json"))
        custom_fields_block = f"\nCustom Fields:\n{custom_fields_str}" if custom_fields_str else ""
        tasks_formatted = "\n".join(project_tasks.get(pid, ["(No associated tasks found)"]))
        block = f"{header}\n{notes}{custom_fields_block}\n\nAssociated Tasks:\n{tasks_formatted}"
        formatted_blocks.append(block)

    if not formatted_blocks:
        return []

    return [f"### Asana Project & Task Context\n\n" + "\n\n".join(formatted_blocks[:500])]  # Limit for safety

def sync_asana_data_once():
    """
    Run a one-time sync of Asana projects and tasks into the local database.
    """
    logger.info("[ASANA SYNC] Starting Asana sync pass.")
    try:
        workspaces_api = asana.WorkspacesApi(asana_api_client)
        workspace_gid = list(workspaces_api.get_workspaces({}))[0]['gid']

        # Fetch all custom fields in the workspace
        custom_fields_api = asana.CustomFieldsApi(asana_api_client)
        custom_fields = custom_fields_api.get_custom_fields_for_workspace(workspace_gid, {})
        with db_connection() as conn:
            c = conn.cursor()
            for cf in custom_fields:
                c.execute('''
                    INSERT OR REPLACE INTO asana_custom_fields
                    (gid, name, type, enum_options_json)
                    VALUES (?, ?, ?, ?)
                ''', (
                    cf.get("gid"),
                    cf.get("name"),
                    cf.get("type"),
                    json.dumps(cf.get("enum_options", []))
                ))
            conn.commit()

        projects = projects_api.get_projects_for_workspace(
            workspace_gid,
            {"opt_fields": "gid,name,notes,created_at,modified_at,custom_fields"}
        )

        with db_connection() as conn:
            c = conn.cursor()

            # Insert projects
            for project in projects:
                custom_fields_json = json.dumps(project.get("custom_fields", []))
                c.execute('''
                    INSERT OR REPLACE INTO asana_projects
                    (gid, name, notes, created_at, modified_at, custom_fields_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    project.get("gid"),
                    project.get("name"),
                    project.get("notes"),
                    project.get("created_at"),
                    project.get("modified_at"),
                    custom_fields_json
                ))

                # Fetch tasks for this project
                tasks = tasks_api.get_tasks_for_project(
                    project.get("gid"),
                    {"opt_fields": "gid,name,notes,assignee,completed,due_on,projects,modified_at,custom_fields"}
                )
                for task in tasks:
                    assignee_gid = task["assignee"]["gid"] if task.get("assignee") else None
                    custom_fields_json = json.dumps(task.get("custom_fields", []))
                    c.execute('''
                        INSERT OR REPLACE INTO asana_tasks
                        (gid, name, notes, assignee, completed, due_on, project_id, modified_at, custom_fields_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        task.get("gid"),
                        task.get("name"),
                        task.get("notes"),
                        assignee_gid,
                        int(task.get("completed", False)),
                        task.get("due_on"),
                        project.get("gid"),
                        task.get("modified_at"),
                        custom_fields_json
                    ))
            conn.commit()
        logger.info("[ASANA SYNC] Completed Asana sync.")
    except Exception as e:
        logger.error(f"[ASANA SYNC] Error during sync: {e}", exc_info=True)

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

FAILED_DELETES = []

def save_or_update_message(msg, channel_info, action="upsert", source="event"):
    """
    Insert, update, or mark a message as deleted in the database.

    Args:
        msg (dict): Slack message dictionary (must include 'ts').
        channel_info (dict): Channel info dictionary (must include 'id').
        action (str): "upsert" for insert/update, "delete" for hard-delete.
        source (str): Event source, for logging/audit.
    """
    ts = msg.get('ts')
    channel_id = channel_info['id']
    max_retries = 5
    retry_delay = 0.5  # seconds

    if action == "delete":
        # logger.info(f"[DEBUG] Attempting to hard delete message ts={ts} from channel={channel_id}.")
        for attempt in range(max_retries):
            try:
                with db_connection() as conn:
                    c = conn.cursor()
                    c.execute('DELETE FROM messages WHERE ts = ? AND channel_id = ?', (ts, channel_id))
                    c.execute('DELETE FROM responded_messages WHERE ts = ?', (ts,))
                    # Add more DELETEs for related tables if needed
                    conn.commit()
                logger.info(f"[DB] Hard deleted message ts={ts} from channel {channel_id} and related tables.")
                break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    logger.warning(f"[DB] Database is locked, retrying delete (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"[DB] Error during delete: {e}", exc_info=True)
                    break
        else:
            logger.error(f"[DB] Failed to delete message ts={ts} after {max_retries} retries due to database lock.")
            FAILED_DELETES.append((ts, channel_id))
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
            # logger.info(f"[DEBUG] Attempting to upsert message ts={ts} in channel={channel_id}.")
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
    if not message:
        return False
    message_lower = message.lower()
    return any(message_lower.split().count(trigger) > 0 for trigger in SEARCH_TRIGGERS)

def extract_search_terms_and_instruction(message, logger=None):
    extraction_prompt = f"""Extract from this message:
    1. Keywords to search for (comma-separated)
    2. Instruction for presenting results (after ";"). The instruction should include the user's requested action verb (e.g., 'provide', 'summarize', 'list', etc.) and be as complete as possible.

    Example response for "Find docs about AI and summarize key points"

    "AI, documentation; summarize key points"

    Message: "{message}

    Response:"""

    try:
        response = generate_response(extraction_prompt, safe=True).strip()
        if ";" in response:
            terms_part, instruction = response.split(";", 1)
            keywords = [t.strip() for t in terms_part.split(",")]
            # Filter out generic stopwords
            keywords = [k for k in keywords if k.lower() not in GENERIC_STOPWORDS and len(k) > 1]
            return keywords, instruction.strip()
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

def clear_all_history():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages")
        c.execute("DELETE FROM last_seen")
        conn.commit()
    logger.info("[BACKFILL] Cleared all message and last_seen history for full backfill.")
    
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

def count_messages(keywords=None):
    with db_connection() as conn:
        c = conn.cursor()
        if keywords:
            query = "SELECT COUNT(*) FROM messages WHERE deleted = 0 AND (" + " OR ".join(["text LIKE ?"] * len(keywords)) + ")"
            params = [f'%{term}%' for term in keywords]
            c.execute(query, params)
        else:
            c.execute("SELECT COUNT(*) FROM messages WHERE deleted = 0")
        return c.fetchone()[0]

def count_asana_projects_and_tasks():
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM asana_projects")
        projects = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM asana_tasks")
        tasks = c.fetchone()[0]
    return projects, tasks

def periodic_backfill(interval=BACKFILL_INTERVAL):
    """Periodic backfill scheduler with auto-vacuum, null entry cleanup, and retry of failed deletes."""
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

            # Retry failed hard deletes
            global FAILED_DELETES
            if FAILED_DELETES:
                logger.info(f"[BACKFILL] Retrying {len(FAILED_DELETES)} failed deletes...")
            for ts, channel_id in FAILED_DELETES[:]:
                try:
                    with db_connection() as conn:
                        c = conn.cursor()
                        c.execute('DELETE FROM messages WHERE ts = ? AND channel_id = ?', (ts, channel_id))
                        c.execute('DELETE FROM responded_messages WHERE ts = ?', (ts,))
                        # Add more DELETEs for related tables if needed
                        conn.commit()
                    FAILED_DELETES.remove((ts, channel_id))
                    logger.info(f"[BACKFILL] Successfully retried and hard deleted message ts={ts} from channel={channel_id} and related tables.")
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        logger.warning(f"[BACKFILL] Database still locked for ts={ts}, channel={channel_id}. Will retry later.")
                        continue
                    else:
                        logger.error(f"[BACKFILL] Error retrying delete for ts={ts}, channel={channel_id}: {e}", exc_info=True)
                        continue

        except Exception as e:
            logger.error(f"[BACKFILL] Periodic backfill error: {e}", exc_info=True)
        
        time.sleep(interval)

def backfill_and_process_mentions():
    """Backfill missed messages and process mentions (with pagination), skipping already logged messages."""
    try:
        logger.info("[BACKFILL] Starting backfill process")

        # Determine if this is a fresh backfill by checking if any rows exist at all
        with db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM last_seen")
            is_fresh_start = (c.fetchone()[0] == 0)
        logger.info(f"[BACKFILL] Fresh backfill: {is_fresh_start}")

        channels = client.conversations_list(types="public_channel")['channels']

        for channel in channels:
            channel_id = channel['id']
            if not channel.get('is_member', False):
                continue

            last_ts = get_last_seen(channel_id)
            if not last_ts or last_ts in ('', '0'):
                last_ts = '0'
                is_fresh_start = True
            else:
                is_fresh_start = False

            logger.debug(f"[BACKFILL] Backfilling {channel['name']} from {last_ts}")

            cursor = None
            latest_seen = last_ts

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

                    if not is_fresh_start and float(ts) <= float(last_ts):
                        logger.debug(f"[BACKFILL] Skipping message ts={ts} (already seen, last_ts={last_ts})")
                        continue

                    if "has joined the channel" in msg.get('text', '').lower():
                        continue

                    if not msg.get('text'):
                        logger.debug(f"[BACKFILL] Skipping message ts={ts} due to empty or None text.")
                        continue

                    with db_connection() as conn:
                        c = conn.cursor()
                        c.execute("SELECT 1 FROM messages WHERE ts=? AND channel_id=?", (ts, channel_id))
                        if c.fetchone():
                            # logger.debug(f"[DEBUG] Already in DB: ts={ts} channel={channel_id}")
                            continue

                    # logger.info(f"[DEBUG] Logging NEW message ts={ts}")
                    store_message_with_check(msg, channel)

                    # Update last_seen after logging
                    with db_connection() as conn:
                        c = conn.cursor()
                        c.execute('INSERT OR REPLACE INTO last_seen (channel_id, last_ts) VALUES (?, ?)', (channel_id, ts))
                        conn.commit()

                    latest_seen = ts

                    if f"<@{BOT_USER_ID}>" in msg.get('text', ''):
                        logger.info(f"[MENTION] Found mention in backfill: ts={ts}")
                        log_mention(msg.get('user'), channel_id, msg.get('text'))
                        try:
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
                            ), client, logger)
                        except Exception as e:
                            logger.warning(f"[MENTION] Failed to process mention during backfill: {e}")

                    mark_as_responded(ts)

                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break

            if latest_seen != last_ts:
                with db_connection() as conn:
                    c = conn.cursor()
                    c.execute('INSERT OR REPLACE INTO last_seen (channel_id, last_ts) VALUES (?, ?)', (channel_id, latest_seen))
                    conn.commit()

        logger.info("[BACKFILL] Backfill complete")

        logger.info("[BACKFILL] Starting one-time Asana sync after backfill.")
        sync_asana_data_once()

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

    # Prevent the bot from responding to its own messages
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
    keywords, instruction, user_id, channel_id, say, client, logger,
    source="command", super_search=False, notice_ts=None
):
    logger.info(
        f"[SEARCH][{source.upper()}] Handling request - "
        f"Keywords: {keywords} | Instruction: {instruction} | SuperSearch: {super_search}"
    )

    # Dynamically get Asana context for these keywords
    asana_context = get_asana_context_blocks(keywords) if keywords else []

    if super_search:
        raw_results = search_messages_local_flat(limit=None)
    else:
        raw_results = search_messages_across_channels(
            keywords, logger=logger
        )

    results = filter_search_messages(raw_results)

    if not results and not asana_context:
        say(f"No messages or Asana projects found for: {', '.join(keywords) if keywords else 'all messages'}", channel=channel_id)
        return

    sources = results
    slack_chunks = [
        [format_message_for_context(r) for r in chunk]
        for chunk in chunk_list(results, CHUNK_SIZE)
    ]

    all_chunks = slack_chunks.copy()
    if asana_context:
        all_chunks.append(asana_context)

    all_snippets = [s for chunk in all_chunks for s in chunk]

    # Calculate total number of chunks for logging
    num_messages = len(results)
    total_chunks = math.ceil(num_messages / CHUNK_SIZE) if num_messages else 1

    prompt_template = (
        f"{BOT_PERSONALITY_PROMPT}\n\n"
        f"User request:\n"
        f"\"\"\"{{instruction}}\"\"\"\n\n"
        f"Context from Slack messages and other sources:\n"
        f"\"\"\"{{context_block}}\"\"\"\n\n"
        "Using only the context above, craft a direct, complete, and helpful answer to the user's request. "
        "Do not critique or summarize the context block itself. Do not refer to the context block directly. "
        "Respond as if you are answering the user directly."
    )

    if len(all_snippets) > CHUNK_SIZE * total_chunks:
        logger.warning("[SEARCH] Result too large for full summarization, falling back to chunk summaries.")
        summaries = []
        for i, chunk in enumerate(all_chunks):
            logger.info(
                f"[SEARCH][CHUNK] Summarizing chunk {i+1}/{total_chunks} with {len(chunk)} messages."
            )
            chunk_prompt = prompt_template.format(
                instruction=instruction,
                context_block=chr(10).join(chunk)
            )
            summary = generate_response(chunk_prompt, safe=True)
            summaries.append(summary)
            time.sleep(DELAY_BETWEEN_CALLS)

        final_prompt = prompt_template.format(
            instruction=instruction,
            context_block=chr(10).join(summaries)
        )
        final_summary = generate_response(final_prompt, safe=True)
    else:
        prompt = prompt_template.format(
            instruction=instruction,
            context_block=chr(10).join(all_snippets)
        )
        final_summary = generate_response(prompt, safe=True)

    sources_note = f"_{len(sources)} sources found._" if sources else "_No relevant sources found._"

    context_header = (
        f"*Requestor:* <@{user_id}>\n"
        f"*Search Type:* {'Super Search' if super_search else 'Standard'}\n"
        f"*Keywords:* {', '.join(keywords) if keywords else '(all messages)'}\n"
        f"*Context:* {instruction}\n\n"
        f"{sources_note}\n\n"
    )

    summary_post = client.chat_postMessage(
        channel=channel_id,
        text=context_header + "\n" + format_for_slack(final_summary or ""),
    )
    parent_ts = summary_post["ts"]

    # Remove the announcement message if present
    if notice_ts:
        try:
            client.chat_delete(channel=channel_id, ts=notice_ts)
        except Exception as e:
            logger.warning(f"Failed to delete public notice message: {e}")


@app.command("/dev-clean-sweep")
def handle_dev_channels_delete_recent(ack, respond, command, client, logger):
    ack()
    try:
        # Fetch all public and private channels, paginated
        channels = []
        cursor = None
        while True:
            resp = client.conversations_list(types="public_channel,private_channel", limit=1000, cursor=cursor)
            channels.extend(resp["channels"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        # Only channels starting with dev and bot is a member
        dev_channels = [c for c in channels if c["name"].startswith("dev") and c.get("is_member", False)]
        logger.info(f"dev_channels found: {[c['name'] for c in dev_channels]}")
        if not dev_channels:
            respond("No channels found starting with 'dev'.")
            return

        deleted = []
        for channel in dev_channels:
            channel_id = channel["id"]
            total_deleted = 0
            total_skipped = 0
            cursor = None
            while True:
                try:
                    resp = client.conversations_history(channel=channel_id, limit=1000, cursor=cursor)
                    messages = resp.get("messages", [])
                    if not messages:
                        break
                    for msg in messages:
                        ts = msg.get("ts")
                        # Delete threaded replies first, if any
                        if msg.get("reply_count", 0) > 0 and msg.get("thread_ts"):
                            thread_cursor = None
                            while True:
                                try:
                                    thread_resp = client.conversations_replies(
                                        channel=channel_id,
                                        ts=msg["thread_ts"],
                                        cursor=thread_cursor,
                                        limit=200
                                    )
                                    replies = thread_resp.get("messages", [])[1:]  # skip parent
                                    for reply in replies:
                                        reply_ts = reply.get("ts")
                                        while True:
                                            try:
                                                client.chat_delete(channel=channel_id, ts=reply_ts)
                                                total_deleted += 1
                                                break
                                            except SlackApiError as e:
                                                error = e.response["error"]
                                                if error == "ratelimited":
                                                    retry_after = int(e.response.headers.get("Retry-After", 30))
                                                    logger.warning(f"Rate limited. Sleeping for {retry_after} seconds...")
                                                    time.sleep(retry_after)
                                                    continue
                                                else:
                                                    logger.warning(f"Could not delete thread reply {reply_ts} in {channel['name']}: {error}")
                                                    total_skipped += 1
                                                    break
                                            except Exception as e:
                                                logger.warning(f"Could not delete thread reply {reply_ts} in {channel['name']}: {e}")
                                                total_skipped += 1
                                                break
                                    thread_cursor = thread_resp.get("response_metadata", {}).get("next_cursor")
                                    if not thread_cursor:
                                        break
                                except Exception as e:
                                    logger.warning(f"Failed to fetch/delete thread replies in {channel['name']}: {e}")
                                    break
                        # Delete the parent/top-level message
                        while True:
                            try:
                                client.chat_delete(channel=channel_id, ts=ts)
                                total_deleted += 1
                                break
                            except SlackApiError as e:
                                error = e.response["error"]
                                if error == "ratelimited":
                                    retry_after = int(e.response.headers.get("Retry-After", 30))
                                    logger.warning(f"Rate limited. Sleeping for {retry_after} seconds...")
                                    time.sleep(retry_after)
                                    continue
                                else:
                                    logger.warning(f"Could not delete message {ts} in {channel['name']}: {error}")
                                    total_skipped += 1
                                    break
                            except Exception as e:
                                logger.warning(f"Could not delete message {ts} in {channel['name']}: {e}")
                                total_skipped += 1
                                break
                    cursor = resp.get("response_metadata", {}).get("next_cursor")
                    if not cursor:
                        break
                except Exception as e:
                    logger.warning(f"Failed to fetch/delete in {channel['name']}: {e}")
                    break
            deleted.append(f"{channel['name']} (deleted: {total_deleted}, skipped: {total_skipped})")
        respond(
            f"Delete complete. Results:\n" +
            "\n".join(deleted)
        )
    except Exception as e:
        logger.error(f"Error deleting all dev messages: {e}", exc_info=True)
        respond("An error occurred while deleting messages.")

@app.command("/ai-asana-query")
def handle_asana_query_command(ack, respond, command, say, logger, client):
    ack()
    instruction = command.get("text", "").strip()
    user_id = command.get("user_id")
    channel_id = command.get("channel_id")

    if not instruction:
        say(f":warning: <@{user_id}> Please include an instruction, like `/ai-asana-query what tasks are due soon?`")
        return

    say(
        f":clipboard: <@{user_id}> requested an Asana query: _'{instruction}'_. This may take a moment. Results will be posted here."
    )

    logger.info(f"[AI ASANA QUERY] Received request from user {user_id} in channel {channel_id}: {instruction}")

    try:
        logger.info("[AI ASANA QUERY] Connecting to database and retrieving records.")

        # Step 1: Load and convert DB rows to plain dicts
        with db_connection() as conn:
            c = conn.cursor()
            tasks = [dict(row) for row in c.execute("SELECT * FROM asana_tasks").fetchall()]
            projects = [dict(row) for row in c.execute("SELECT * FROM asana_projects").fetchall()]
        logger.info(f"[AI ASANA QUERY] Retrieved {len(projects)} projects and {len(tasks)} tasks.")

        # Step 2: Group tasks by project
        from collections import defaultdict
        project_tasks = defaultdict(list)
        for task in tasks:
            project_id = task.get("project_id")
            task_str = f"- {task.get('name')} [Due: {human_time(task.get('due_on'))} | Completed: {bool(task.get('completed'))} | Last Updated: {human_time(task.get('modified_at'))}]"
            notes = task.get("notes", "")
            clean_notes = re.sub(r'https?://\S+', '[link removed]', notes)  # remove URLs
            clean_notes = re.sub(r'\s+', ' ', clean_notes).strip()
            if clean_notes:
                task_str += f"\n  Notes: {clean_notes[:1000]}..."
            project_tasks[project_id].append(task_str)
        logger.info("[AI ASANA QUERY] Grouped tasks by project.")

        # Step 3: Format blocks per project
        formatted_blocks = []
        for project in projects:
            pid = project.get("gid")
            header = f"*Project: {project.get('name')}*\nLast Updated: {human_time(project.get('modified_at'))}"
            notes = f"Notes: {project.get('notes')[:1000]}." if project.get("notes") else ""
            tasks_formatted = "\n".join(project_tasks.get(pid, ["(No associated tasks found)"]))
            block = f"{header}\n{notes}\n\nAssociated Tasks:\n{tasks_formatted}"
            formatted_blocks.append(block)
        logger.info("[AI ASANA QUERY] Assembled formatted blocks for prompt.")

        # Step 4: Build AI prompt
        prompt = (
            f"The user has asked: \"{instruction}\"\n\n"
            f"{BOT_PERSONALITY_PROMPT}\n\n"
            f"Below is a structured list of Asana projects and their associated tasks. Please analyze this context and respond with insight and clarity.\n\n"
            f"{'-'*40}\n\n"
            + "\n\n".join(formatted_blocks[:100])  # Limit to 100 projects
        )
        logger.info("[AI ASANA QUERY] Prompt ready, sending to Gemini.")

        # Step 5: Generate AI response
        response_text = generate_response(prompt)
        logger.info("[AI ASANA QUERY] Response generated successfully.")

        formatted = format_for_slack(response_text)
        
        response_header = (
        f"*Requestor:* <@{user_id}>\n"
        f"*Search Type:* Asana Query\n"
        f"*Request:* {instruction}\n\n"
        f"_Reviewed {len(projects)} projects and {len(tasks)} tasks_\n\n"
        )
        full_response = response_header + formatted
        say(full_response, channel=channel_id)

    except Exception as e:
        logger.error(f"[AI ASANA QUERY] Error processing request: {e}", exc_info=True)
        say("Sorry, something went wrong while processing the Asana query.")

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

    # Count messages, projects, and tasks
    num_messages = count_messages(keywords)
    num_projects, num_tasks = count_asana_projects_and_tasks()
    total_sources = num_messages + num_projects + num_tasks

    # Post a public notice and capture the ts
    notice = client.chat_postMessage(
        channel=channel_id,
        text=(
            f":mag: <@{user_id}> requested a search for *{', '.join(keywords) if keywords else '(all messages)'}* "
            f"with instruction: _'{instruction}'_. Searching {num_messages} messages, {num_projects} Asana projects, and {num_tasks} Asana tasks. ({total_sources} total sources). This may take a few minutes. Results will be posted here."
        )
    )
    notice_ts = notice["ts"]

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
    
    # After posting the final result, delete the notice
    try:
        client.chat_delete(channel=channel_id, ts=notice_ts)
    except Exception as e:
        logger.warning(f"Failed to delete public notice message: {e}")

@app.command("/ai-super-search")
def handle_super_search_command(ack, respond, command, say, logger, client):
    ack()
    text = command.get("text", "").strip()
    user_id = command.get("user_id")
    channel_id = command.get("channel_id")

    instruction = text if text else "Summarize the findings"

    # Count all messages, projects, and tasks
    num_messages = count_messages()
    num_projects, num_tasks = count_asana_projects_and_tasks()
    total_sources = num_messages + num_projects + num_tasks

    say(
        f":mag: <@{user_id}> requested a *Super Search* with instruction: _'{instruction}'_. Searching {num_messages} messages, {num_projects} Asana projects, and {num_tasks} Asana tasks. ({total_sources} total sources). This may take up to 5 minutes. Results will be posted here."
    )

    handle_search_request(
        keywords=[],
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

def search_messages_across_channels(search_terms, logger=None):
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

        # Count messages, projects, and tasks
        num_messages = count_messages(keywords)
        num_projects, num_tasks = count_asana_projects_and_tasks()
        total_sources = num_messages + num_projects + num_tasks

        # Send "searching" message to the channel (public)
        try:
            notice = client.chat_postMessage(
                channel=channel_id,
                text=(
                    f":mag: <@{user_id}> requested a search for *{', '.join(keywords) if keywords else '(all messages)'}* "
                    f"with instruction: _'{instruction}'_. Searching {num_messages} messages, {num_projects} Asana projects, and {num_tasks} Asana tasks ({total_sources} total sources). This may take a few minutes. Results will be posted here."
                )
            )
            notice_ts = notice["ts"]
        except Exception as e:
            logger.error(f"Failed to send search message: {e}")
            notice_ts = None

        handle_search_request(
            keywords=keywords,
            instruction=instruction,
            user_id=user_id,
            channel_id=channel_id,
            say=say,
            client=client,
            logger=logger,
            source="mention",
            notice_ts=notice_ts,  # Pass the ts
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
    return (
        f"{BOT_PERSONALITY_PROMPT}\n\n"
        f"User request:\n\n"
        f'"""{context["primary_message"]}"""\n\n'
        f"Channel context:\n\n"
        f'"""{context["previous_messages"]}"""\n\n'
        "Respond clearly, confidently, and in a format that fits the request. Prioritize completion."
    )

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
    atexit.register(lambda: print(" [EXIT] Interrupted by user."))
    
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