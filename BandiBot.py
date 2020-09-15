import discord
import youtube_dl
import audioread
import asyncio
import nacl
import os
import time
from discord.ext import commands, tasks
from discord import FFmpegPCMAudio
from discord.voice_client import VoiceClient
from discord.utils import get
from youtube_dl import YoutubeDL
from random import choice

client = commands.Bot(command_prefix = '-')


@client.command()
async def on_ready():
    print('BandiBot está listo :D')

@client.command()
async def saludo(ctx):
    await ctx.send(f'Hello! I\'m BandiBot :smile:')

@client.command()
async def join(ctx):
    if (ctx.author.voice):
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send('Joined :thumbsup:')
    else: 
        await ctx.send("Bro... You are not in a voice channel :neutral_face:")

@client.command()
async def leave(ctx):
    if (ctx.voice_client): 
        await ctx.guild.voice_client.disconnect() 
    else: 
        await ctx.send("I'm not even in a voice channel :man_facepalming: ")
        

  

client.run('NzUzODY1Njg3OTYwNjQ5NzQ5.X1saIg.d0X--DnFqs8Ju_xqAstbwx_11og')