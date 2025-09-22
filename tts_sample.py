from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import scipy



processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# from text
text_inputs = processor(text = "Hello, my name is Bharath I would like to order a cake", src_lang="eng", return_tensors="pt")
audio_array_from_text = model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()

sample_rate = model.config.sampling_rate
scipy.io.wavfile.write("out_from_text.wav", rate=sample_rate, data=audio_array_from_text)