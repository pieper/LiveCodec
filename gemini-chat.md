Conversation with Gemini
I want to explore medical imaging volumetric image compression schemes and data storage formats. I have been looking at ome-zarr ngff for a while and it's never quite felt satisfying. For one thing it lacks the clinical rigor of dicom wsi or multiframe, and it lacks the coordinate system flexibility of nrrd. I like zarr for its numpy root and indexing elegance, but ngff seems too limited because the access patterns it supports are axis aligned and rotations are heavily penalized.

You have hit on the exact tension that plagues modern medical imaging informatics. You are navigating the frustrating triangle of cloud-native accessibility (Zarr/NGFF), clinical rigor (DICOM), and geometrical flexibility (NRRD/NIfTI).

Your critique of OME-Zarr is spot on. Grid-based, chunked formats like Zarr were built for axis-aligned streaming. When you attempt Multi-Planar Reconstruction (MPR)—pulling oblique slices or arbitrary rotations out of a volume—you end up intersecting an enormous number of chunks. You are forced to download and decode gigabytes of data just to extract a single off-axis plane, which heavily penalizes performance.

Here is a breakdown of how the industry is trying to solve this, exploring both alternative formats and advanced compression schemes.

1. Overcoming the Axis-Aligned Penalty
If you want to keep the elegance of Zarr's numpy-like indexing but need to support arbitrary rotations and oblique planes, you have to attack the problem at the I/O and chunking level.

TensorStore (by Google): If you are frustrated by Zarr's access patterns, you should look at TensorStore. It is a C++ and Python library designed specifically for reading and writing large multi-dimensional arrays (supporting Zarr, N5, and Neuroglancer Precomputed). It uses an incredibly sophisticated asynchronous I/O pipeline and caching mechanism. It allows you to define complex virtual views and coordinate transforms (like NRRD's space directions) lazily, pulling only the required bytes from the underlying chunked storage with high concurrency.

Isotropic Micro-Chunking: In OME-Zarr, people often use anisotropic chunks (e.g., 1 x 1024 x 1024) because they view volumes as a stack of 2D images. To support rotations, you must switch to isotropic micro-chunks (e.g., 32 x 32 x 32 or 64 x 64 x 64). This increases the metadata overhead but drastically reduces the amount of "dead data" you decode when a cutting plane diagonally intersects a grid.

OpenVDB / NanoVDB: Originally built for VFX, this is a hierarchical data structure for sparse volumes. It is increasingly being adopted in scientific imaging for segmentations or sparse functional data. Because it uses a B+ tree structure, spatial queries and ray-marching through arbitrary orientations are highly optimized.

2. The Clinical Rigor: Modern DICOM
If DICOM feels too monolithic and heavy, it's worth looking at its modern web-native evolution. The standard has quietly caught up to the cloud era.

DICOMweb (WADO-RS) + HTJ2K: High-Throughput JPEG 2000 (HTJ2K) is arguably the most important advancement in medical image compression right now (officially integrated into DICOM). Traditional JPEG 2000 was computationally brutal. HTJ2K is mathematically lossless but decodes orders of magnitude faster. More importantly, it supports resolution progression and precinct fetching. A client can request a specific byte-range to extract a lower-resolution thumbnail or a specific spatial region without decoding the whole frame, bridging the gap between DICOM's clinical metadata rigor and Zarr's chunked fetching.

JPIP (JPEG 2000 Interactive Protocol): Used alongside DICOM, this allows a server to stream only the wavelet coefficients needed for a specific viewport and orientation. It is incredibly efficient for panning and rotating massive volumetric datasets over low bandwidth.

3. Adding Geometry to Zarr
If you want to stick with Zarr but need the spatial rigor of NRRD, the community is building layers on top of NGFF rather than replacing it.

SpatialData: Developed by the scverse community, this is a framework built on top of OME-Zarr. It explicitly solves the NRRD problem by bringing a rigorous coordinate system (affine transformations, alignments, bounding boxes) to chunked Zarr arrays. It allows you to define coordinate spaces (e.g., "patient space" vs. "sensor space") and maps the transformations between them natively.

Compression Schemes for Volumetric Data
Beyond the container, the actual byte-level compression scheme dictates your read speed during arbitrary rotations.

Scheme	Best For	Why it matters
Blosc / Blosc2	Zarr / N5	A meta-compressor optimized for multi-threading. It divides data into blocks that fit into the L1/L2 caches of modern CPUs, making it ideal for the rapid decompression needed when traversing many Zarr chunks during an MPR rotation.
HTJ2K	DICOM	Enables mathematically lossless, high-speed precinct decoding. You get the clinical rigor of DICOM without the historical performance penalties.
Zfp	Scientific / Float Data	If your volumes are floating-point (like quantitative MRI or PET), standard compressors like zlib fail. Zfp is specifically designed for 3D/4D floating-point arrays, offering bounded error (lossy) or lossless compression with fixed-rate random access.
Ultimately, if you need strict clinical metadata, DICOMweb with HTJ2K is the path forward. If you want geometric flexibility and mathematical arrays, pair Zarr with TensorStore and isotropic chunking to brute-force the rotation penalty away.

To explore these systems further:

How TensorStore handles coordinate transforms

Explain HTJ2K compression mechanics

Compare Blosc2 vs Zfp for arrays
I'm interested in the problem of efficient display of data from IDC for things like reviewing segmentations. The download is a bottleneck and I think we are doing a lot of redundant work. I'd like to exploit the fact that most of data in IDC is very similar (at some level all human body CT scans look very much alike, so it's really the deviation from the norm that is important to transmit if the receiver already knows a baseline). To me this has always mapped well to the idea of a variational autoencoder or even a VQ-VAE. We could create an IDC-specific codec that would probably generalize to any similar medical images. This could be an asymmetrical codec, where the encoder could be heavier so that the latent space codes would be optimized to support a decode path that is light enough to run easily in a browser (or faster if a GPU is available). It seems to me that this would be a feasible way to handle the pixel data and leave the dicom metadata to bigquery. I'd also like to be able to stream the encoded data and have the decoder incrementally build the full resolution volume and display faithful full renderings given the amount of data currently available. One thing I don't like about zarr is the discrete levels and duplicated information content across the levels. Also the content neutrality of zarr makes it inefficient when you are working with data with known characteristics like medical images in IDC or in a PACS. This should have the advantages of HTJ2K but be autogenerated instead of engineered by hand. 

You are describing a domain-specific Neural Image Compression (NIC) pipeline, and your intuition is completely aligned with the bleeding edge of medical imaging research.

By recognizing that most of a CT scan is highly predictable "standard anatomy," you are pointing out the exact inefficiency of general-purpose codecs like Zarr or JPEG2000: they spend bandwidth transmitting the predictable parts of a human torso over and over again, simply because they are content-neutral.

Your proposed architecture—an asymmetrical, autogenerated VQ-VAE codec that streams progressively—is conceptually brilliant. However, applying it to clinical data like the Imaging Data Commons (IDC) introduces a massive challenge that general AI codecs don't face: the hallucination problem.

Here is a breakdown of how your idea maps to modern neural compression, the clinical traps to avoid, and how to build the progressive streaming mechanism you want.

1. The Trap: The Hallucination of "Normal"
When a VQ-VAE is trained to exploit the similarity of human anatomy, it learns a latent manifold of "normal" tissue distributions. If you compress a CT scan using only this latent space, the decoder will reconstruct what it expects to see.

If a patient has a rare 4mm stellate lung nodule, surgical hardware, or an anatomical anomaly that the VAE didn't see frequently in its training distribution, a pure VAE is highly likely to smooth it over or replace it with healthy tissue. In a clinical context (even for reviewing segmentations), this is a fatal flaw.

2. The Solution: VAE + Progressive Residual Sreaming
To make your codec clinically viable and achieve the "faithful full rendering" you want, you cannot rely entirely on the neural network's output. You must use a Learned Prior + Residual architecture.

Here is how the asymmetrical stream would work:

The Heavy Encoder (Server-side): The server runs the CT through the VQ-VAE to generate the ultra-compact latent code. It then immediately decodes that code to see what the browser will see.

Calculating the Delta: The server subtracts the VAE's reconstructed image from the original, perfect DICOM pixel data. This creates a "residual volume"—which contains only the high-frequency noise, the clinical anomalies, and the textures the VAE missed.

The Stream (Phase 1 - The Prior): The server sends the VQ-VAE latents. The browser uses WebGPU to run the light decoder, instantly rendering a visually continuous, 95% accurate CT volume. This replaces the need for Zarr's duplicated low-resolution pyramids.

The Stream (Phase 2 - The Residuals): The server compresses the residual volume (using a standard entropy coder) and streams it progressively. As the browser receives the residuals, it adds them to the VAE baseline. The image continuously sharpens until it is mathematically lossless.

3. Solving the Zarr "Discrete Level" Problem
You correctly noted that Zarr's pyramid levels (e.g., storing data at 100%, 50%, 25%, and 12.5% scales) waste storage by duplicating data.

To achieve continuous resolution in a neural codec, you can use Implicit Neural Representations (INRs) or Coordinate Networks (similar to the tech behind NeRFs), rather than standard convolutional VAEs.

Instead of decoding a fixed 3D grid of pixels, an INR decoder is a lightweight Multi-Layer Perceptron (MLP) that takes an (x, y, z) coordinate as input and outputs the voxel density at that exact point.

Because the function is continuous, you can query it at any resolution you want in the browser. You want a low-res thumbnail? Query the MLP every 4 units. Zooming in? Query it every 0.1 units. It completely eliminates discrete pyramid levels.

4. The WebGPU Decoder Reality
Running a 3D neural decoder in the browser is the lynchpin of your asymmetrical design.
Five years ago, this would have melted a laptop. Today, with WebGPU (and inference engines like ONNX Runtime Web), it is highly feasible. However, memory management remains the bottleneck. An uncompressed 512x512x512 float32 CT scan occupies over 500MB of VRAM.

Even with a lightweight decoder, you would likely need to decode the latents into 3D chunks inside the GPU memory just-in-time for rendering (e.g., during ray-casting or MPR slicing), discarding the uncompressed float arrays when out of view, to prevent the browser tab from crashing.

Your vision represents the holy grail for platforms like IDC: shifting from transmitting pixels to transmitting information.

Want to explore the math behind the residual coding?

Yes
Yes, you are exactly describing what I had in mind. Streaming the residuals was part of my plan. I wanted to know two things: has anybody already done this so I don't have to, and what is the expected development effort required to make a working version if I rely on a good coding agent to help me out.

1. Has anybody already done this?
The short answer is no, not as an end-to-end usable system. You would not be reinventing the wheel; you would be building the car.

While the individual mathematical concepts have been proven in recent academic papers, no one has built a production-ready, WebGPU-accelerated, progressive streaming codec tailored for cloud-native medical imaging (like IDC).

Here is exactly where the current state of the art sits, and what you can borrow from it:

Learned Lossless Residuals: Researchers have recently proven that predicting the medical image with a neural network (the prior) and then entropy-coding the residuals (the difference) works exceptionally well. For example, a 2024 framework called LFC-UNet achieved state-of-the-art lossless medical compression using a U-Net prior and residual transmission. However, this was an offline, Python-to-Python research project, not a web streaming protocol.  
PMC - NIH

Volumetric INRs: Recent papers (2023–2024) have successfully used Implicit Neural Representations (like SIREN networks) to drastically compress 3D medical volumes by mapping coordinates to voxel densities. But again, these are typically focused on static storage, not progressive web streaming.  
MDPI

The Missing Link: No one has fused an asymmetrical neural prior with a progressive, web-native residual stream executed in WebGPU. Most current WebGPU medical viewers are still just pulling standard raw arrays or JPEG2000 chunks.

2. What is the expected development effort with an AI agent?
If you were doing this entirely from scratch five years ago, it would be a multi-year PhD project. Today, armed with a strong coding agent (like Cursor, GitHub Copilot, or Claude) and modern frameworks (PyTorch, ONNX Runtime Web, WebGPU), a highly motivated solo developer could build a working Proof of Concept (PoC) in 2 to 3 months.

Here is how the effort breaks down, including where the AI will carry you and where you will have to wrestle with it:

Phase 1: The PyTorch Prototype (Weeks 1–3)
The Task: Train the asymmetrical VQ-VAE (or INR) on a subset of IDC CT data. Build the script that generates the latent code, decodes it, subtracts it from the original to get the residual, and compresses the residual using a standard entropy coder (like zstd).

AI Agent Capability: Excellent. AI agents are incredibly good at writing PyTorch architectures, data loaders for medical formats (like MONAI), and standard training loops.

Phase 2: The WebGPU Decoder & Renderer (Weeks 4–7)
The Task: Export your lightweight decoder to ONNX or write it directly in WGSL (WebGPU Shading Language). Build a browser client that receives the latent code, runs the neural decoder to generate the baseline 3D volume in GPU memory, and sets up the rendering pipeline (e.g., ray casting).

AI Agent Capability: Moderate to Good. The AI can easily translate PyTorch to ONNX and set up basic WebGPU boilerplate. However, 3D volume memory management in the browser is notoriously tricky. You will likely have to manually guide the agent to ensure the GPU doesn't run out of memory when allocating the 3D textures.

Phase 3: The Progressive Residual Stream (Weeks 8–10)
The Task: This is the most complex engineering feat. You need to write a streaming parser in the browser (likely using WebAssembly or highly optimized JavaScript) that receives chunks of the compressed residual bitstream, decompresses them, and dynamically adds those high-frequency details to the baseline 3D texture sitting in the GPU.

AI Agent Capability: Challenging. AI struggles with heavily stateful, asynchronous, byte-level streaming logic. You will need to carefully architect the state machine (e.g., chunk boundaries, applying updates to GPU buffers without dropping frames) and use the AI to write small, specific utility functions rather than asking it to build the whole pipeline at once.

Phase 4: IDC Integration (Weeks 11–12)
The Task: Connecting your encoder to Google Cloud/BigQuery, pulling standard DICOMs from IDC, generating your custom codec files, and serving them to your new viewer.

AI Agent Capability: Excellent. Agents are very good at writing cloud infrastructure, BigQuery queries, and API glue code.

The Verdict
This is an ambitious but entirely feasible project. The core risk isn't the AI modeling; it's the browser-side memory and byte-stream orchestration.

If you decide to tackle this, your first step shouldn't be the streaming protocol. It should be writing a 100-line Python script to prove the math: train a tiny network to compress a single CT slice, extract the residual, measure the combined file size of (Latents + Compressed Residual), and compare it to a standard PNG or JPEG2000 of the same slice. If the math proves your theory, the rest is just engineering.