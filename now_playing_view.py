"""Now Playing embed + button controls.

Posts a visual playback display in chat when a song starts. The big visual
identity lives in a custom-generated banner image (see banner.py). The embed
itself provides a live timer in the footer and updates its state as the queue
progresses.

State machine:
  - Song playing   → banner image + live timer footer + active buttons
  - Next song      → banner regenerated with new track, timer resets
  - Queue empty    → no image, "Queue finished" message, buttons disabled
"""

import asyncio
import io
import logging
from typing import Optional

import discord

from banner import generate_banner
from music import voice_manager

logger = logging.getLogger(__name__)

EMBED_COLOR = 0x4A148C
EMPTY_COLOR = 0x2C2C3A  # dimmer color for the queue-empty state
TIMER_INTERVAL = 5     # seconds between footer timer edits


def _format_duration(seconds) -> str:
    """Format seconds as M:SS or H:MM:SS. Returns '—:—' if unknown."""
    if seconds is None or seconds <= 0:
        return "—:—"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _build_playing_embed(
    queue_size: int,
    requested_by: str,
    duration_str: str = "0:00 / —:—",
) -> discord.Embed:
    embed = discord.Embed(color=EMBED_COLOR)
    embed.set_author(name="✦  Now Playing")
    embed.set_image(url="attachment://now_playing.png")
    embed.set_footer(
        text=f"⏱ {duration_str}                             🎵 {queue_size} in queue\n👤 Requested by {requested_by}"
    )
    return embed


def _build_empty_embed() -> discord.Embed:
    """Build the embed for the queue-empty state."""
    embed = discord.Embed(
        description="## Queue finished\nAdd more songs by mentioning me.",
        color=EMPTY_COLOR,
    )
    embed.set_author(name="✦  BandiBot")
    return embed


class NowPlayingView(discord.ui.View):
    """The button row attached to a Now Playing embed.

    Owns the timer update task and handles state transitions when songs
    change or the queue empties.
    """

    def __init__(self, guild: discord.Guild, player):
        super().__init__(timeout=None)
        self.guild = guild
        self.player = player
        self.message: Optional[discord.Message] = None
        self._timer_task: Optional[asyncio.Task] = None

    def start_updates(self):
        """Start the background timer task."""
        if self._timer_task:
            self._timer_task.cancel()
        self._timer_task = asyncio.create_task(self._timer_loop())

    async def stop_updates(self):
        """Cancel the timer task."""
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    async def _timer_loop(self):
        try:
            while True:
                await asyncio.sleep(TIMER_INTERVAL)
                if not self.message:
                    break
                track = self.player.current
                if not track:
                    break
                if track.paused_at is not None:
                    continue

                elapsed = self.player.elapsed_seconds
                duration_str = f"{_format_duration(int(elapsed))} / {_format_duration(track.duration)}"

                try:
                    embed = _build_playing_embed(
                        queue_size=len(self.player.queue),
                        requested_by=track.requested_by,
                        duration_str=duration_str,
                    )
                    await self.message.edit(embed=embed)
                except discord.NotFound:
                    self.message = None
                    break
                except Exception as e:
                    logger.error(f"[now_playing] timer update failed: {e}")
        except asyncio.CancelledError:
            pass

    async def on_track_changed(self, track, queue_size: int):
        if not self.message:
            return
        try:
            png_bytes = await generate_banner(
                track.title, track.artist, track.thumbnail
            )
        except Exception as e:
            logger.error(f"[now_playing] banner regen failed: {e}")
            return

        duration_str = f"0:00 / {_format_duration(track.duration)}"
        embed = _build_playing_embed(
            queue_size=queue_size,
            requested_by=track.requested_by,
            duration_str=duration_str,
        )
        file = discord.File(fp=io.BytesIO(png_bytes), filename="now_playing.png")
        try:
            await self.message.edit(attachments=[file], embed=embed, view=self)
            self.start_updates()
        except Exception as e:
            logger.error(f"[now_playing] track change edit failed: {e}")

    async def on_queue_empty(self):
        """Called when the queue runs out. Transitions to empty state."""
        await self.stop_updates()

        # Disable all buttons
        for item in self.children:
            item.disabled = True

        if not self.message:
            return

        try:
            await self.message.edit(
                attachments=[],
                embed=_build_empty_embed(),
                view=self,
            )
        except Exception as e:
            logger.error(f"[now_playing] queue empty edit failed: {e}")

    # ---- Row 1: playback controls ----

    @discord.ui.button(emoji="⏮", style=discord.ButtonStyle.secondary, row=0)
    async def previous_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("⏮ Previous — coming soon", ephemeral=True)

    @discord.ui.button(emoji="⏯", style=discord.ButtonStyle.secondary, row=0)
    async def pause_resume_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        player = voice_manager.get_player(self.guild)
        if not player.is_connected:
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)
            return
        if player.voice_client.is_paused():
            result = await voice_manager.resume(self.guild)
        else:
            result = await voice_manager.pause(self.guild)
        await interaction.response.send_message(result, ephemeral=True)

    @discord.ui.button(emoji="⏭", style=discord.ButtonStyle.secondary, row=0)
    async def skip_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        result = await voice_manager.skip(self.guild)
        await interaction.response.send_message(result, ephemeral=True)

    @discord.ui.button(emoji="⏹", style=discord.ButtonStyle.danger, row=0)
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        result = await voice_manager.stop(self.guild)
        await interaction.response.send_message(result, ephemeral=True)

    # ---- Row 2: extras (placeholders for now) ----

    @discord.ui.button(emoji="🔉", style=discord.ButtonStyle.secondary, row=1)
    async def volume_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("🔉 Volume — coming soon", ephemeral=True)

    @discord.ui.button(emoji="🔁", style=discord.ButtonStyle.secondary, row=1)
    async def loop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("🔁 Loop — coming soon", ephemeral=True)

    @discord.ui.button(emoji="🔀", style=discord.ButtonStyle.secondary, row=1)
    async def shuffle_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("🔀 Shuffle — coming soon", ephemeral=True)

    @discord.ui.button(emoji="📋", style=discord.ButtonStyle.secondary, row=1)
    async def queue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        result = await voice_manager.get_queue(self.guild)
        await interaction.response.send_message(result, ephemeral=True)


async def post_now_playing(
    channel: discord.TextChannel,
    player,
    *,
    title: str,
    artist: Optional[str],
    duration_seconds: Optional[int],
    queue_size: int,
    requested_by: str,
    thumbnail_url: Optional[str] = None,
) -> Optional["NowPlayingView"]:
    try:
        png_bytes = await generate_banner(title, artist, thumbnail_url)
    except Exception as e:
        logger.error(f"[now_playing] banner generation failed: {e}")
        return None

    duration_str = f"0:00 / {_format_duration(duration_seconds)}"
    file = discord.File(fp=io.BytesIO(png_bytes), filename="now_playing.png")
    embed = _build_playing_embed(
        queue_size=queue_size,
        requested_by=requested_by,
        duration_str=duration_str,
    )
    view = NowPlayingView(channel.guild, player)

    try:
        message = await channel.send(file=file, embed=embed, view=view)
    except Exception as e:
        logger.error(f"[now_playing] failed to post: {e}")
        return None

    view.message = message
    view.start_updates()
    player._now_playing_view = view
    player.now_playing_message = message

    return view


async def update_now_playing_queue(player, queue_size: int):
    msg = player.now_playing_message
    if not msg or not player.current:
        return

    try:
        track = player.current
        embed = _build_playing_embed(
            queue_size=queue_size,
            requested_by=track.requested_by,
        )
        await msg.edit(embed=embed)
    except Exception as e:
        logger.error(f"[now_playing] failed to update queue count: {e}")
        player.now_playing_message = None