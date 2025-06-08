from cog import BasePredictor, Input, Path
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self):
        print("🔄 Lade FLUX Modell...")
        
        # Basis FLUX-Modell laden
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("🔄 Lade AndrinK Modell von Hugging Face...")
        
        try:
            # Dein Modell direkt von Hugging Face downloaden
            model_path = hf_hub_download(
                repo_id="MonsterMMORPG/AndrinK",
                filename="FLUX/AndrinK_FLUX-000100.safetensors"
            )
            
            # Custom Weights laden
            custom_weights = load_file(model_path)
            self.pipe.transformer.load_state_dict(custom_weights, strict=False)
            print("✅ AndrinK Modell geladen!")
            
        except Exception as e:
            print(f"⚠️ Fehler: {e}")
            print("📝 Nutze Standard FLUX-Modell")
        
        self.pipe.enable_model_cpu_offload()
        print("✅ Bereit!")

    def predict(self, prompt: str = Input()) -> Path:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=1024,
            width=768
        ).images[0]
        
        output_path = "/tmp/image.png"
        image.save(output_path)
        return Path(output_path)
