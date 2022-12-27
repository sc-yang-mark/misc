import tempfile
import torchaudio
import soundfile
import librosa

def convert_to_16k_use_soundfile(audio_path, output_path):
    wav, sr = librosa.load(audio_path)
    wav_16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    soundfile.write(output_path, wav_16, 16000)

def convert_to_16k_use_torchaudio(audio_path, output_path):
    wav, sr = torchaudio.load(audio_path)

    effects = []
    if sr != 16000:
        effects.append(['rate', str(16000)])
    if wav.shape[0] > 1:  # multi-channel
        effects.append(['channels', '1'])
    if len(effects) > 0:
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
    torchaudio.save(output_path, wav, sr, bits_per_sample=16)

# 一般使用方式
convert_to_16k_use_soundfile('test.wav', 'test-16k.wav')

# 轉檔並寫入至暫存檔
# 在 Windows 需要將 delete 設定為 False
temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
convert_to_16k_use_soundfile('test.wav', temp_file.name)
