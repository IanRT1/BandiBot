"""
music/attachments.py

Discord audio attachment ingestion for BandiBot's music system.

Text-channel uploads are treated as first-class playable tracks when their
extension matches a supported audio format. Attachments bypass the LLM tool
flow entirely: they are inspected, converted into Track objects, queued under
the requesting user, and played through the same GuildPlayer path as YouTube
tracks.

Pipeline:
  Discord attachment → extension filter → byte fetch → mutagen metadata parse
  → Track construction → VoiceManager queue → Discord confirmation reply

Metadata support:
  - MP3       → ID3 title, artist, duration, APIC cover art
  - FLAC      → Vorbis comments, duration, embedded pictures
  - M4A/AAC   → MP4 atoms, duration, cover art
  - OGG/OPUS  → Vorbis comments, duration, FLAC PICTURE block cover art

Fallback behavior:
  If metadata or cover art extraction fails, the attachment still queues using
  the filename as the title. Metadata failures are logged but never block
  playback.
"""

import base64
import io
import logging
import os
import struct

import aiohttp
import discord
from mutagen.flac import FLAC
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis

from bot.utils import clean_username
from music.player import voice_manager
from music.tracks import Track

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".opus", ".ogg", ".m4a", ".aac"}


def get_audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """Return attachments whose extension is in _AUDIO_EXTENSIONS."""
    result = []
    for attachment in message.attachments:
        ext = os.path.splitext(attachment.filename)[1].lower()
        if ext in _AUDIO_EXTENSIONS:
            result.append(attachment)
    return result


async def _extract_audio_metadata(url: str, ext: str) -> dict:
    """Download attachment bytes and extract title, artist, duration, cover art via mutagen."""
    result = {
        "title": None,
        "artist": None,
        "duration": None,
        "thumbnail_bytes": None,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.warning(f"[music] metadata fetch returned {resp.status}")
                    return result
                data = await resp.read()

        buf = io.BytesIO(data)

        if ext == ".flac":
            f = FLAC(buf)
            result["title"] = f.get("title", [None])[0]
            result["artist"] = f.get("artist", [None])[0] or f.get("performer", [None])[0]
            result["duration"] = int(f.info.length) if f.info else None
            if f.pictures:
                result["thumbnail_bytes"] = f.pictures[0].data

        elif ext == ".mp3":
            buf.seek(0)
            f = MP3(buf)
            result["duration"] = int(f.info.length) if f.info else None
            try:
                buf.seek(0)
                tags = ID3(buf)
                tit2 = tags.get("TIT2")
                tpe1 = tags.get("TPE1")
                result["title"] = str(tit2) if tit2 else None
                result["artist"] = str(tpe1) if tpe1 else None
                apic = tags.getall("APIC")
                if apic:
                    result["thumbnail_bytes"] = apic[0].data
            except Exception:
                pass

        elif ext in (".m4a", ".aac"):
            buf.seek(0)
            f = MP4(buf)
            result["duration"] = int(f.info.length) if f.info else None
            tags = f.tags or {}
            title = tags.get("\xa9nam")
            artist = tags.get("\xa9ART")
            result["title"] = title[0] if title else None
            result["artist"] = artist[0] if artist else None
            covr = tags.get("covr")
            if covr:
                result["thumbnail_bytes"] = bytes(covr[0])

        elif ext in (".ogg", ".opus"):
            buf.seek(0)
            f = OggVorbis(buf)
            result["title"] = f.get("title", [None])[0]
            result["artist"] = f.get("artist", [None])[0]
            result["duration"] = int(f.info.length) if f.info else None
            pic_b64 = f.get("metadata_block_picture", [None])[0]
            if pic_b64:
                try:
                    pic_data = base64.b64decode(pic_b64)
                    offset = 4
                    mime_len = struct.unpack(">I", pic_data[offset:offset + 4])[0]
                    offset += 4 + mime_len
                    desc_len = struct.unpack(">I", pic_data[offset:offset + 4])[0]
                    offset += 4 + desc_len + 16
                    img_len = struct.unpack(">I", pic_data[offset:offset + 4])[0]
                    offset += 4
                    result["thumbnail_bytes"] = pic_data[offset:offset + img_len]
                except Exception as e:
                    logger.warning(f"[music] ogg cover art parse failed: {e}")

    except Exception as e:
        logger.error(f"[music] metadata extraction failed: {e}")

    return result


async def handle_audio_attachments(
    message: discord.Message,
    attachments: list[discord.Attachment],
) -> None:
    """Queue audio attachments directly into the music system, bypassing LLM."""
    requester = message.author

    if not requester.voice or not requester.voice.channel:
        await message.reply(
            "Únete a un canal de voz primero para que pueda reproducir el archivo.",
            mention_author=False,
        )
        return

    guild = message.guild
    voice_channel = requester.voice.channel
    player = voice_manager.get_player(guild)
    requested_by = clean_username(requester.nick, requester.name)

    async with voice_manager._get_play_lock(guild.id):
        if not player.text_channel:
            player.text_channel = message.channel

        await player.connect(voice_channel)

        queued_titles = []
        is_busy = player.is_playing or (
            player.is_connected and player.voice_client.is_paused()
        )

        for attachment in attachments:
            ext = os.path.splitext(attachment.filename)[1].lower()
            filename_title = os.path.splitext(attachment.filename)[0]

            logger.info(f"[music] extracting metadata from {attachment.filename!r}")
            meta = await _extract_audio_metadata(attachment.url, ext)

            title = meta["title"] or filename_title
            artist = meta["artist"]
            duration = meta["duration"]
            thumbnail_bytes = meta["thumbnail_bytes"]

            track = Track(
                title=title,
                stream_url=attachment.url,
                requested_by=requested_by,
                webpage_url=attachment.url,
                duration=duration,
                thumbnail=None,
                thumbnail_bytes=thumbnail_bytes,
                artist=artist,
                resolved=True,
            )
            player.queue.append(track)
            queued_titles.append(title)
            logger.info(
                f"[music] queued attachment: {title!r} | "
                f"artist={artist!r} | duration={duration}s | "
                f"cover={'yes' if thumbnail_bytes else 'no'}"
            )

        if not is_busy:
            player.play_next()

    if len(queued_titles) == 1:
        if is_busy:
            await message.reply(f"Agregado a la cola: **{queued_titles[0]}**", mention_author=False)
        else:
            await message.reply(f"Reproduciendo: **{queued_titles[0]}**", mention_author=False)
    else:
        titles_str = "\n".join(f"• {t}" for t in queued_titles)
        await message.reply(
            f"Agregados {len(queued_titles)} archivos a la cola:\n{titles_str}",
            mention_author=False,
        )
