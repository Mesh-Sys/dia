import torch
from dia.model import Dia

# Select device: CPU
device = torch.device("cpu")
print(f"Using device: {device}")

target_model = "nari-labs/Dia-1.6B-0626"

#model = Dia.from_pretrained(target_model, compute_dtype="float16")
#model = Dia.from_pretrained(target_model, compute_dtype="float16", device=device)
model = Dia.from_pretrained(target_model, compute_dtype="float32", device=device)
print(f"Loaded model - '{target_model}'")

# You should put the transcript of the voice you want to clone
# We will use the audio created by running simple.py as an example.
# Note that you will be REQUIRED TO RUN simple.py for the script to work as-is.
#clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
#clone_from_text = "[S1] Huge thanks to Asmongold for somehow finding my video and then reacting to it, and to everyone who's here now because of that. I'm legitimately like brand new to YouTube, and I genuinely did not expect this video to get really any attention, much less blow up now the way that it has, but here we are. So before all this, it really was just me and my 300 Spartans holding the line in the comments section and now it appears we've been overrun by roaches, which by the way, you guys."
#clone_from_text = "[S1] Huge thanks to Asmongold for somehow finding my video and then reacting to it, and to everyone who's here now because of that. I'm legitimately like brand new to YouTube, and I genuinely did not expect this video to get really any attention, much less blow up now the way that it has, but here we are. So before all this, it really was just me and my 300 Spartans holding the line in the comments section and now it appears we've been overrun by roaches, which by the way, you guys [S2] Thanks to you all"
#clone_from_text = "[S1] Huge thanks to Asmongold for somehow finding my video and then reacting to it, and to everyone who's here now because of that. I'm legitimately like brand new to YouTube, and I genuinely did not expect this video to get really any attention, much less blow up now the way that it has, but here we are. So before all this, it really was just me and my 300 Spartans holding the line in the comments section and now it appears we've been overrun by roaches, which by the way, you guys"
#clone_from_audio = "simple.mp3"
#clone_from_text = "[S1] This is not the end of Doge, but really the beginning. My time as a special government employee necessarily had to end. It was a limited time thing. It's 134 days, I believe, which adds in a few days. So that comes with a time limit."
#clone_from_audio = "girl.mp3"
clone_from_text = "[S1] Huge thanks to Asmongold for somehow finding my video and then reacting to it, and to everyone who's here now because of that. I'm legitimately like brand new to YouTube."
clone_from_audio = "girl_10s.mp3"

# For your custom needs, replace above with below and add your audio file to this directory:
# clone_from_text = "[S1] ... [S2] ... [S1] ... corresponding to your_audio_name.mp3"
# clone_from_audio = "your_audio_name.mp3"

# Text to generate
#text_to_generate = "[S1] Hello, how are you? [S2] I'm good, thank you. [S1] What's your name? [S2] My name is Dia. [S1] Nice to meet you. [S2] Nice to meet you too."
#text_to_generate = "[S1] Hello, how are you? [S2] I'm good, thank you. [S1] What's your name? [S2] My name is Dia. [S1] Nice to meet you. [S2] Nice to meet you too."
#text_to_generate = "[S1] And I genuinely did not expect this video to get really any attention, much less blow up now the way that it has. [S1] But here we are. [S1] So before all this. [S1] It really was just me and my 300 Spartans holding the line in the comments section. [S1] And now it appears we've been overrun by roaches. [S1] which by the way, you guys."
text_to_generate = [
    "[S1] And I genuinely did not expect this video to get really any attention, much less blow up now the way that it has.",
    #"[S1] But here we are.",
    #"[S1] So before all this.",
    #"[S1] It really was just me and my 300 Spartans holding the line in the comments section.",
    #"[S1] And now it appears we've been overrun by roaches.",
    #"[S1] which by the way, you guys.",
]

# It will only return the audio from the text_to_generate
#output = model.generate(
#    clone_from_text + text_to_generate,
#    audio_prompt=clone_from_audio,
#    use_torch_compile=False,
#    verbose=True,
#    cfg_scale=4.0,
#    temperature=1.8,
#    top_p=0.90,
#    cfg_filter_top_k=50,
#)
from pydub import AudioSegment
#x = 0
#for input_text in text_to_generate:
#    output = model.generate(
#        (clone_from_text if x == 0 else text_to_generate[x - 1]) + input_text,
#        audio_prompt=clone_from_audio if x == 0 else f"voice_clone_{x - 1}.mp3",
#        use_torch_compile=False,
#        verbose=True,
#        cfg_scale=4.0,
#        temperature=1.8,
#        top_p=0.90,
#        cfg_filter_top_k=50,
#    )
#    model.save_audio(f"voice_clone_{x}.mp3", output)
#
#processed_sound = AudioSegment.from_mp3(clone_from_audio)
#for i in range(len(text_to_generate)):
#    processed_sound = processed_sound.append(AudioSegment.from_mp3(f"voice_clone_{i}.mp3"))
#processed_sound.export("voice_clone_merged.mp3", format="mp3")
#    x += 1
#processed_sound = AudioSegment.from_mp3(clone_from_audio)
#for i in range(len(text_to_generate)):
#    processed_sound = processed_sound.append(AudioSegment.from_mp3(f"voice_clone_{i}.mp3"))
#processed_sound.export("voice_clone_merged.mp3", format="mp3")

#text_heap = clone_from_text
#processed_sound = AudioSegment.from_mp3(clone_from_audio)
#x = 0
#for input_text in text_to_generate:
#    output = model.generate(
#        text_heap + input_text,
#        audio_prompt=clone_from_audio if x == 0 else f"voice_clone_merged_{x - 1}.mp3",
#        use_torch_compile=False,
#        verbose=True,
#        cfg_scale=4.0,
#        temperature=1.8,
#        top_p=0.90,
#        cfg_filter_top_k=50,
#    )
#    text_heap += input_text
#    model.save_audio(f"voice_clone_{x}.mp3", output)
#    processed_sound = processed_sound.append(AudioSegment.from_mp3(f"voice_clone_{x}.mp3"))
#    processed_sound.export(f"voice_clone_merged_{x}.mp3", format="mp3")
#    x += 1

text_heap = clone_from_text
processed_sound = AudioSegment.from_mp3(clone_from_audio)
x = 0
for input_text in text_to_generate:
    print(f"Generating - `{input_text}`")
    output = model.generate(
        text_heap + input_text,
        audio_prompt=clone_from_audio if x == 0 else f"voice_clone_merged_{x - 1}.mp3",
        use_torch_compile=False,
        verbose=True,
        #cfg_scale=4.0,
        cfg_scale=1.0,
        #temperature=1.8,
        temperature=1.2,
        top_p=0.90,
        cfg_filter_top_k=50,
    )
    text_heap += input_text
    model.save_audio(f"voice_clone_{x}.mp3", output)
    processed_sound = processed_sound.append(AudioSegment.from_mp3(f"voice_clone_{x}.mp3"))
    processed_sound.export(f"voice_clone_merged_{x}.mp3", format="mp3")
    x += 1
print("Done")