import subprocess
from pathlib import Path

def get_compressonator_path() -> Path:
    env_path = Path(__file__).parent.parent / ".env"
    default_path = Path("C:\\Compressonator_4.5.52\\bin\\CLI\\compressonatorcli.exe")
    
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("COMPRESSONATOR_PATH="):
                        # Extract value after =
                        value = line.split("=", 1)[1].strip()
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        if value:
                            return Path(value)
        except Exception:
            pass
            
    return default_path

class bcn:

    #constructor
    def __init__(
        self,
        input_image,
        output_image,
        format="BC1",
        quality=0.75,
        use_gpu=True
    ):
        self.cli_path = get_compressonator_path()
        self.input_image = Path(input_image)
        self.output_image = Path(output_image)
        self.format = format
        self.quality = quality
        self.use_gpu = use_gpu

    #the subprocess function that runs the command
    def run(self):
        if not (0.05 <= self.quality <= 1.0):
            raise ValueError("quality must be between 0.05 and 1.0")

        cmd = [
            str(self.cli_path),
            "-fd", self.format,
            "-Quality", str(self.quality),
        ]

        if self.use_gpu:
            cmd += ["-EncodeWith", "GPU"]

        cmd += [
            "-nomipmap",
            str(self.input_image),
            str(self.output_image),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return result
