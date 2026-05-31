"""
bot/tool_schemas.py

LLM function/tool schemas for BandiBot.

These definitions describe the tools the model is allowed to call. They are
sent to OpenAI by the text and voice prompt layers, then executed by
bot/tool_executor.py after the model chooses a tool and arguments.

Boundary:
  This module is declarative only. It does not call OpenAI, mutate Discord
  state, touch the music queue, or execute tool side effects.
"""

MUSIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Play a single song or audio in the user's current voice channel. "
                "Pass the user's exact words as the query — do NOT interpret, translate, or try to guess the correct song title. "
                "Accepts either a YouTube URL or a free-text search query. "
                "If something is already playing, the new track is added to the queue. "
                "Use this whenever the user asks to play or queue a SINGLE song."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's exact words as the search query. Do not interpret or modify.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_bulk",
            "description": (
                "Queue multiple songs at once. Use this when the user asks to add several songs, "
                "provides a list of songs, or pastes a YouTube playlist URL. "
                "For text lists, each item becomes a separate search query. "
                "For a YouTube playlist URL, pass it as the single item in the list. "
                "ONLY include songs explicitly requested in the CURRENT message. "
                "Do NOT include songs from previous messages or that are already playing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of search queries or URLs. Each item is a separate song. "
                            "For a YouTube playlist, pass the playlist URL as the only item."
                        ),
                    },
                    "is_playlist": {
                        "type": "boolean",
                        "description": "True if the input is a YouTube playlist URL, False for text search queries.",
                    },
                },
                "required": ["queries", "is_playlist"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_track",
            "description": "Move a song in the queue to a different position. Can reference the track by its current position number or by name/partial name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_position": {
                        "type": "integer",
                        "description": "Current position of the track in the queue (1-based). Optional if track_name is provided.",
                    },
                    "to_position": {
                        "type": "integer",
                        "description": "Target position in the queue (1-based).",
                    },
                    "track_name": {
                        "type": "string",
                        "description": "Name or partial name of the track to move. Used if from_position is not provided.",
                    },
                },
                "required": ["to_position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_track",
            "description": (
                "Remove one or more songs from the queue without affecting playback. "
                "Can reference tracks by position number(s), by name/partial name, "
                "or by recency. Does NOT affect the currently playing track."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "positions": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of queue positions to remove (1-based). Use this for one or multiple positions.",
                    },
                    "track_name": {
                        "type": "string",
                        "description": "Name or partial name of the track to remove. Also handles relative references like 'last song'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clip_audio",
            "description": (
                "Record and send the last 30 seconds of voice channel audio as an mp3 file in chat. "
                "Use when the user asks to clip, record, or save what was just said or played "
                '(eg. "Bandibot Clip that!").'
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "join_voice",
            "description": (
                "Join the voice channel that the user is currently in. "
                "Use this when the user asks the bot to join, come, or enter their voice channel. "
                "Works even if no music is requested."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_track",
            "description": "Skip the currently playing track and move to the next in queue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_music",
            "description": "Pause the currently playing track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_music",
            "description": "Resume a paused track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_music",
            "description": "Stop playback and clear the entire queue. Bot stays in the voice channel.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "leave_voice",
            "description": "Disconnect from the voice channel entirely.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "now_playing",
            "description": "Get the title of the currently playing track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_queue",
            "description": "List the upcoming tracks in the queue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

GET_MEMBER_ACTIVITY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_member_activity",
            "description": (
                "Get real-time information about who is currently online, "
                "what they are playing, who is in voice channels, and their "
                "roles and permissions. Use this when the user asks about "
                "server members, who is online, who is in voice, or what "
                "people are doing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

GET_SERVER_INFO_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_server_info",
            "description": (
                "Get information about this Discord server from its history and lore document. "
                "Use this when the user asks something about the server — its history, rules, "
                "events, or any other server-specific knowledge. "
                "Pass the user's question so the response can be tailored to what was asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about the server, used to contextualize the response.",
                    }
                },
                "required": ["question"],
            },
        },
    }
]

ALL_TOOLS = MUSIC_TOOLS + GET_MEMBER_ACTIVITY_TOOL + GET_SERVER_INFO_TOOL
