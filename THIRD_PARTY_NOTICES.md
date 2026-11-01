# Third-party components and source materials

The repository MIT license applies to the RxnOptBench authors' original software. It does not relicense external packages, model weights, API services, source publications, benchmark records or archived model outputs; consult the component-specific licenses and rights notice for those materials.

No third-party dependency source code or model weights are bundled in this release. Dependencies installed from the requirements files retain their own terms. In particular:

- **PyMuPDF / MuPDF**: the optional document-rendering and screening stages use PyMuPDF, which upstream distributes under AGPL or commercial terms. See the [official licensing description](https://pymupdf.readthedocs.io/en/latest/about.html). Installing or combining it with this code does not convert PyMuPDF to MIT.
- **DECIMER-Segmentation**: the molecule-detection stage calls the separately installed DECIMER package. See the [upstream source](https://github.com/Kohulan/DECIMER-Image-Segmentation), [MIT license](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/LICENSE), and [citation](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/CITATION.cff). No DECIMER implementation or weights are included here.
- **Other installed dependencies**: Pillow, OpenCV, filelock, NumPy, OpenAI Python SDK, RDKit, requests, SciPy, Matplotlib, scikit-learn, PyTorch, Transformers, Accelerate, and vLLM retain their upstream licenses. Requirements files identify package constraints; consult the distribution installed in your environment for its exact notices.

The extraction workflow expects source documents/images supplied lawfully by the user and, for OCSR alignment, externally generated recognition results. The release does not supply publisher PDFs, supporting-information documents or page images, or an OCSR model implementation/weights. Citation of a source article does not grant rights in that article or its figures.
