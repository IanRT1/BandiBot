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
                "Reason about the user's request and pass a clean YouTube search query, not the raw command. "
                "Remove command words such as play, queue, put on, pon, reproduce, or add. "
                "Keep the song title, artist, featured artist, version/remix/live/remaster clues, and album clues when useful. "
                "If the user gives a plausible song title plus artist, preserve those words literally. "
                "Song titles can contain command-like words such as stop, pause, play, skip, or start from scratch; "
                "when the user says play/pon/reproduce plus those words, treat them as the requested title. "
                "Do not substitute a more famous song by the same artist. "
                "If the user provides a YouTube video ID, pass that exact ID unchanged. "
                "Correct obvious speech-to-text confusions only when the intended song is clear from context, but do not invent a different song. "
                "Accepts either a YouTube URL or a cleaned free-text search query. "
                "For YouTube radio links like watch?v=...&list=RD... or start_radio=1, play the video URL as a single song. "
                "If something is already playing, the new track is added to the queue. "
                "Use this whenever the user asks to play or queue a SINGLE song."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Cleaned YouTube search query containing only the likely title/artist/version terms. "
                            "Do not include wake words, user names, or command verbs. "
                            "When unsure, prefer the literal requested title and artist over a guessed catalog match. "
                            "A bare 11-character YouTube video ID must be passed exactly as written."
                        ),
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
                "Do not use this for YouTube radio links with list=RD... or start_radio=1; those are single-video play_music requests. "
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
                "Use ONLY when the user explicitly asks to remove, delete, clear, drop, or take a track out of the queue. "
                "If the user says to remove/quit the song that is currently playing, use skip_track instead. "
                "Phrases like 'quitar la canción' or 'remove this song' usually mean skip_track when no queue position is given. "
                "Do NOT use for a bare song title or artist mention; bare song requests should usually be play_music. "
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
            "description": (
                "Skip the currently playing track and move to the next in queue. "
                "Use this when the user asks to skip, pass, or remove the song currently playing "
                "(for example, 'quitar la canción', 'quita esta canción', or 'remove this song')."
            ),
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
            "description": (
                "Stop playback and clear the entire queue. Bot stays in the voice channel. "
                "Use ONLY when the CURRENT message explicitly asks to stop playback, stop music, or clear the queue. "
                "Do NOT use for a play request whose title contains command-like words, "
                "such as 'play start from scratch the game'."
            ),
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
                "Use this only when relevant server lore was not already provided in the context. "
                "It retrieves the server history, rules, events, or other server-specific knowledge. "
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

WEB_SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web and return a concise answer with sources. "
                "Use this for current, changing, or externally verifiable information: "
                "news, weather, prices, schedules, public facts, product details, "
                "recent events, or questions you cannot answer reliably from memory. "
                "Do not use this for BandiBot/server lore, music playback, or ordinary conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete natural-language question to search for.",
                    }
                },
                "required": ["question"],
            },
        },
    }
]

ALL_TOOLS = MUSIC_TOOLS + GET_MEMBER_ACTIVITY_TOOL + GET_SERVER_INFO_TOOL + WEB_SEARCH_TOOL


def tools_without_server_info():
    """Return tools for prompts whose server lore was already retrieved."""
    return [
        tool for tool in ALL_TOOLS
        if tool["function"]["name"] != "get_server_info"
    ]


def tools_without_context_lookups():
    """Remove lore/activity lookups when local lore already answers the query."""
    excluded = {"get_server_info", "get_member_activity"}
    return [
        tool for tool in ALL_TOOLS
        if tool["function"]["name"] not in excluded
    ]
