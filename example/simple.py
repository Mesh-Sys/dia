from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

#text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
text = "[S1] Huge thanks to Asmongold for somehow finding my video and then reacting to it, and to everyone who's here now because of that. I'm legitimately like brand new to YouTube, and I genuinely did not expect this video to get really any attention, much less blow up now the way that it has, but here we are. So before all this, it really was just me and my 300 Spartans holding the line in the comments section and now it appears we've been overrun by roaches, which by the way, you guys."

output = model.generate(
    text,
    use_torch_compile=False,
    verbose=True,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)

model.save_audio("simple.mp3", output)
