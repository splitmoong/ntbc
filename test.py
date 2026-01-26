from src.bcn import bcn

# Create BC1 compression
job = bcn(
    input_image="images/donut_basecolour.jpg",
    output_image="donut_bc1.dds",
    format="BC1",
    quality=0.85,
)

# Run compression
result = job.run()


#check result
if result.returncode != 0:
    print("Compression failed")
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
else:
    print("Compression succeeded")
