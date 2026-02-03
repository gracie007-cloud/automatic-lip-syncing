import argparse
import json
import os
import re
import subprocess
import sys

# Mapping Rhubarb shapes to LazyKH shapes
RHUBARB_MAP = {
    'A': 'm',   # Closed (P, B, M)
    'B': 't',   # Consonants
    'C': 'a',   # Open
    'D': 'a',   # Wide open
    'E': 'u',   # Rounded
    'F': 'au',  # Puckered
    'G': 'f',   # F/V
    'H': 'y',   # L
    'X': 'm'    # Idle/Closed
}

def parse_annotated_text(text):
    emotions_map = {
        "explain": 0, "happy": 1, "sad": 2, "angry": 3, "confused": 4, "rq": 5
    }
    
    words = []
    # (word_index, type, value, line_index, paragraph_index)
    tags = [] 
    
    current_word_idx = 0
    current_line_idx = 0
    current_para_idx = 0
    
    # Split by lines first to track line and para indices
    lines = text.split('\n')
    for l_idx, line in enumerate(lines):
        # Paragraph detection (empty line indicates new paragraph)
        if not line.strip() and l_idx > 0:
            current_para_idx += 1
            
        # Tokens in this line
        tokens = re.findall(r'<[^>]+>|\[[^\]]+\]|\S+', line)
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                emotion_name = token[1:-1].lower()
                if emotion_name in emotions_map:
                    tags.append((current_word_idx, 'emotion', emotions_map[emotion_name], l_idx, current_para_idx))
            elif token.startswith('[') and token.endswith(']'):
                # In LazyKH, the image/topic tag just marks the line.
                # The videoDrawer will re-parse the line to find the [topic].
                tags.append((current_word_idx, 'image', l_idx, l_idx, current_para_idx))
            else:
                clean_word = re.sub(r'[^\w\s]', '', token)
                if clean_word:
                    words.append(clean_word)
                    current_word_idx += 1
        
        # At the end of each non-empty line (or even empty ones), image (line_idx) is tracked by l_idx
        # But we only want to record an image change in the schedule if there's a tag or a new line.
        # Actually, scheduler.py adds an image entry for EVERY line.
        tags.append((current_word_idx, 'image_auto', l_idx, l_idx, current_para_idx))

    return words, tags

def run_rhubarb(audio_path, output_json, rhubarb_path):
    print(f"Running Rhubarb on {audio_path}...")
    cmd = [rhubarb_path, "-r", "phonetic", "-f", "json", audio_path, "-o", output_json]
    subprocess.run(cmd, check=True)
    with open(output_json, 'r') as f:
        return json.load(f)

def run_vosk(audio_path, model_path):
    from vosk import Model, KaldiRecognizer
    import wave
    
    print(f"Running Vosk for word timestamps...")
    if not os.path.exists(model_path):
        raise Exception(f"Vosk model not found at {model_path}")
        
    model = Model(model_path)
    
    temp_wav = "temp_vosk.wav"
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", temp_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    wf = wave.open(temp_wav, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    wf.close()
    os.remove(temp_wav)
    
    words_with_time = []
    for res in results:
        if 'result' in res:
            for w in res['result']:
                words_with_time.append(w)
    return words_with_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to exampleVideo/ev (without extension)')
    parser.add_argument('--rhubarb_path', default=r'C:\Users\soray\.github\digital-agents\rhubarb-lip-sync\bin\Rhubarb-Lip-Sync-1.14.0-Windows\rhubarb.exe')
    parser.add_argument('--model_path', default='vosk_model')
    args = parser.parse_args()
    
    base_path = args.input_file
    audio_path = base_path + ".wav"
    text_path = base_path + ".txt"
    schedule_path = base_path + "_schedule.csv"
    
    with open(text_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    plain_words, tags = parse_annotated_text(full_text)
    rhubarb_json = base_path + "_rhubarb.json"
    rhubarb_data = run_rhubarb(audio_path, rhubarb_json, args.rhubarb_path)
    vosk_words = run_vosk(audio_path, args.model_path)
    
    sections = [[] for _ in range(5)]
    
    # Defaults
    sections[0].append("0.000,paragraph,0")
    sections[1].append("0.000,emotion,0")
    sections[2].append("0.000,image,0")
    sections[3].append("0.000,pose,0")
    
    last_l_idx = -1
    last_p_idx = -1
    pose = 0
    
    for word_idx, tag_type, tag_val, l_idx, p_idx in tags:
        # Find timing
        start_time = 0.0
        # If we have words in this line or after, use the timing of the first word found
        if word_idx < len(vosk_words):
            start_time = vosk_words[word_idx]['start']
        elif len(vosk_words) > 0:
            # If we've reached the end of the recognized audio
            start_time = vosk_words[-1]['end']
            
        time_str = f"{start_time:.3f}"
        
        if tag_type == 'emotion':
            sections[1].append(f"{time_str},emotion,{tag_val}")
        elif tag_type == 'image' or tag_type == 'image_auto':
            if l_idx != last_l_idx:
                sections[2].append(f"{time_str},image,{l_idx}")
                # LazyKH changes pose on every new line
                pose = (pose + 1) % 5
                sections[3].append(f"{time_str},pose,{pose}")
                last_l_idx = l_idx
            if p_idx != last_p_idx:
                sections[0].append(f"{time_str},paragraph,{p_idx}")
                last_p_idx = p_idx

    # Phonemes (Rhubarb)
    for cue in rhubarb_data['mouthCues']:
        time_str = f"{cue['start']:.3f}"
        shape = RHUBARB_MAP.get(cue['value'], 'm')
        sections[4].append(f"{time_str},phoneme,{shape}")
        
    with open(schedule_path, 'w', encoding='utf-8') as f:
        for i, section_lines in enumerate(sections):
            # dedup timestamps in same section if they have same value
            unique_lines = []
            last_val = None
            for line in section_lines:
                val = line.split(',')[-1]
                if val != last_val:
                    unique_lines.append(line)
                    last_val = val
            f.write("\n".join(unique_lines) + "\n")
            if i < 4:
                f.write("SECTION\n")
                
    print(f"Schedule created successfully: {schedule_path}")

if __name__ == "__main__":
    main()
