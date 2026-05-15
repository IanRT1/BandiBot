"""
utils.py

Shared utility functions for BandiBot.

Provides helpers for time formatting, member data collection, and server
presence aggregation used across handlers.py and voice_handler.py.

Functions:
  clean_username()        → strip decorative characters (e.g. ・) from display names
  get_current_pst_time()  → current Pacific time formatted as hh:mm AM/PM
  get_current_pst_date()  → current Pacific date formatted as Weekday, Month Day, Year
  get_server_info()       → collect online members, activities, and voice channel state

Privacy:
  Voice channels listed in HIDDEN_VOICE_CHANNEL_IDS are excluded entirely
  from get_server_info() output — neither their names nor their members
  are ever sent to the LLM. The bot sees them via Discord's API but
  filters them before any data leaves the system.

Permission reporting:
  Only a curated subset of permissions (ban_members, manage_channels,
  manage_roles) are surfaced to the LLM to keep context concise and
  avoid exposing the full 40+ Discord permission flag set.

Timezone:
  All time values use America/Los_Angeles (Pacific) regardless of where
  the bot is hosted, since the server community is based there.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import discord

# Build once at module load — cheap, but no point recreating these per call
_PACIFIC = ZoneInfo("America/Los_Angeles")

# Permissions we surface to the model when describing members
IMPORTANT_PERMISSIONS = ("ban_members", "manage_channels", "manage_roles")

# Voice channel IDs whose members must NEVER be exposed to the LLM.
# The bot still sees these channels via Discord's API, but they're filtered
# out of get_server_info() before any data is sent to OpenAI.
HIDDEN_VOICE_CHANNEL_IDS = {
    1373799593128624158,
    1484415471913795584,
}


def clean_username(nick, name):
    """Return nick (with decorative chars stripped) if present, else name."""
    return nick.replace("・", "") if nick else name


def get_current_pst_time():
    """Current Pacific time, formatted as 'hh:mm AM/PM'."""
    return datetime.now(_PACIFIC).strftime("%I:%M %p")


def get_current_pst_date():
    """Current Pacific date, formatted as 'Weekday, Month Day, Year'."""
    return datetime.now(_PACIFIC).strftime("%A, %B %d, %Y")


def _format_permission(perm_name):
    """'ban_members' -> 'Ban Members'."""
    return " ".join(word.capitalize() for word in perm_name.split("_"))


def _days_since(joined_at):
    """Days between a tz-aware datetime and now (UTC)."""
    if joined_at is None:
        return 0
    return (datetime.now(timezone.utc) - joined_at).days


def _get_playing_activity(member):
    """Return the name of the first Game/Playing activity, or None."""
    for activity in member.activities:
        if activity.type == discord.ActivityType.playing:
            return activity.name
    return None


def _get_important_perms(member):
    """Return formatted names of important permissions this member has.

    Iterates the SHORT important-perms tuple instead of all 40+ permission
    flags. Each lookup on member.guild_permissions hits a computed property,
    but discord.py caches the resolved Permissions object per access — and
    we only do 3 lookups instead of 40+.
    """
    perms = member.guild_permissions
    return [
        _format_permission(name)
        for name in IMPORTANT_PERMISSIONS
        if getattr(perms, name)
    ]


def get_server_info(guild):
    """Collect member presence, activity, and voice-channel info for the guild.

    Voice channels listed in HIDDEN_VOICE_CHANNEL_IDS are excluded entirely —
    neither their names nor their members are returned. The LLM never sees
    that they exist.
    """
    online_members = []
    members_playing = []

    for member in guild.members:
        if member.bot:
            continue

        if member.status == discord.Status.online:
            roles = [r.name for r in member.roles if r != guild.default_role]
            online_members.append((
                clean_username(member.nick, member.name),
                roles,
                _days_since(member.joined_at),
                _get_important_perms(member),
            ))

            game = _get_playing_activity(member)
            if game:
                members_playing.append(
                    (clean_username(member.nick, member.name), game)
                )

    voice_channels_info = {}
    for vc in guild.voice_channels:
        if vc.id in HIDDEN_VOICE_CHANNEL_IDS:
            continue
        members_in_vc = [
            clean_username(m.nick, m.name) for m in vc.members if not m.bot
        ]
        if members_in_vc:
            voice_channels_info[vc.name] = members_in_vc

    return {
        "online_count": len(online_members),
        "online_members": online_members,
        "members_playing": members_playing,
        "voice_channels_info": voice_channels_info,
    }