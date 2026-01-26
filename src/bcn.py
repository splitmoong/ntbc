import subprocess
from pathlib import Path

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
        self.cli_path = Path("C:\\Compressonator_4.5.52\\bin\\CLI\\compressonatorcli.exe")
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
    pass