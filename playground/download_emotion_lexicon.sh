#!/usr/bin/env bash
# Run from root of project

mkdir -p emoticon
wget https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip
unzip NRC-Emotion-Lexicon.zip -d emotions
rm NRC-Emotion-Lexicon.zip
tail -n +2 emotions/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt > emotions/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
rm -rf emotions/NRC-Emotion-Lexicon
rm -rf emotions/__MACOSX
