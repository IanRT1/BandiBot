import discord
import pytz
from datetime import datetime

# Define important permissions as constants
IMPORTANT_PERMISSIONS = ["ban_members", "manage_channels", "manage_roles"]


# Function to clean usernames
def clean_username(nick, name):
    """
    Clean a username by removing special characters if present in the nickname.

    Args:
        nick (str): The nickname of the user.
        name (str): The actual name of the user.

    Returns:
        str: The cleaned username.
    """
    return nick.replace("・", "") if nick else name


# Function to get the current time in PST
def get_current_pst_time():
    """
    Get the current time in the Pacific timezone (PST).

    Returns:
        str: The current time in "hh:mm AM/PM" format.
    """
    # Define the Pacific timezone
    pacific = pytz.timezone("US/Pacific")

    # Get the current time in Pacific timezone
    current_pst_time = datetime.now(pacific)

    return current_pst_time.strftime("%I:%M %p")


# Function to get the current date in PST
def get_current_pst_date():
    """
    Get the current date in the Pacific timezone (PST).

    Returns:
        str: The current date in "Day, Month Day, Year" format.
    """
    # Define the Pacific timezone
    pacific = pytz.timezone("US/Pacific")

    # Get the current date in Pacific timezone
    current_pst_date = datetime.now(pacific).date()

    # Format the date to include day of the week, month, day, and year
    return current_pst_date.strftime("%A, %B %d, %Y")


# Function to get server information
def get_server_info(guild):
    """
    Get information about the members and voice channels of a server (guild).

    Args:
        guild (discord.Guild): The Discord server (guild) to get information from.

    Returns:
        dict: A dictionary containing various server information.
    """
    online_members = []
    offline_members = []
    members_playing = []

    for member in guild.members:
        # Skip bot members
        if member.bot:
            continue

        # Categorize members based on their online status
        if member.status == discord.Status.online:
            online_member_info = (
                clean_username(member.nick, member.name),
                [role.name for role in member.roles if role != guild.default_role],
                (
                    datetime.utcnow().replace(tzinfo=member.joined_at.tzinfo)
                    - member.joined_at
                ).days,  # Days since the member joined
                [
                    " ".join(word.capitalize() for word in perm[0].split("_"))
                    for perm in member.guild_permissions
                    if perm[1] and perm[0] in IMPORTANT_PERMISSIONS
                ],
            )
            online_members.append(online_member_info)

            # Check for playing activity for online members
            if member.activity:
                members_playing.append(
                    (clean_username(member.nick, member.name), member.activity.name)
                )
        elif member.status == discord.Status.offline:
            offline_members.append(clean_username(member.nick, member.name))

    # Get members in various voice channels
    voice_channels_info = {}
    for vc in guild.voice_channels:
        members_in_vc = [
            clean_username(member.nick, member.name)
            for member in vc.members
            if not member.bot
        ]
        if members_in_vc:
            voice_channels_info[vc.name] = members_in_vc

    return {
        "online_count": len(online_members),
        "online_members": online_members,
        "offline_members": offline_members,
        "members_playing": members_playing,
        "voice_channels_info": voice_channels_info,
    }
