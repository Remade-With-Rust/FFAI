//! Python bindings for FFai — Diana detection over numpy, PIL and torch arrays.
//!
//! ```python
//! import ffai, numpy as np
//! d = ffai.Detector("n")                 # tier; "rect" geometry by default
//! r = d.detect(np.asarray(pil_image))    # HWC uint8 RGB
//! r.xyxy, r.conf, r.cls, r.names
//! ```
//!
//! # Why this exists
//!
//! Ultralytics accepts a numpy array, a PIL image or a torch tensor directly.
//! FFai already accepted in-memory pixels in Rust — `detect(&ImageBuffer, ..)` —
//! so the gap was never the engine, it was that there was no way to call it from
//! Python at all. This is that, and nothing more: the model, the letterbox and
//! the decode are the same code the CLI runs.
//!
//! # What it accepts, and why the conversion is done HERE
//!
//! Anything numpy can view as a `uint8` array: a numpy array, a PIL image
//! (which exposes `__array_interface__`), or a torch tensor (which exposes
//! `__array__`). Non-arrays are routed through `numpy.asarray` on the Python
//! side rather than asking callers to convert, because "pass me exactly a numpy
//! array" is the kind of API friction that makes people keep using the other
//! library.
//!
//! **RGB is assumed, and that is a real decision.** OpenCV hands out BGR, so
//! `cv2.imread(...)` needs `cv2.cvtColor(..., COLOR_BGR2RGB)` first — same as
//! Ultralytics requires. Silently accepting BGR would shift every colour-
//! dependent detection with no error, so `from_bgr=True` is offered explicitly
//! instead of guessed.

use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

/// One image's detections, shaped like Ultralytics' `Results.boxes`.
#[pyclass]
struct Detections {
    #[pyo3(get)]
    names: Py<PyList>,
    xyxy: Vec<f32>,
    conf: Vec<f32>,
    cls: Vec<u32>,
}

#[pymethods]
impl Detections {
    /// `(N, 4)` float32 boxes in the SOURCE image's pixel coordinates — the
    /// letterbox is already inverted, as it is for the CLI.
    #[getter]
    fn xyxy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.conf.len();
        let a = PyArray2::<f32>::zeros(py, [n, 4], false);
        // SAFETY: freshly allocated, exclusively owned, exactly n*4 elements.
        unsafe {
            a.as_slice_mut()?.copy_from_slice(&self.xyxy);
        }
        Ok(a)
    }

    /// `(N,)` float32 confidences.
    #[getter]
    fn conf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.conf)
    }

    /// `(N,)` uint32 class ids.
    #[getter]
    fn cls<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.cls)
    }

    fn __len__(&self) -> usize {
        self.conf.len()
    }

    /// Ultralytics-style one-liner: `2 persons, 1 tv`.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let names = self.names.bind(py);
        let mut tally: std::collections::BTreeMap<u32, usize> = Default::default();
        for c in &self.cls {
            *tally.entry(*c).or_insert(0) += 1;
        }
        if tally.is_empty() {
            return Ok("Detections()".into());
        }
        let parts: Vec<String> = tally
            .iter()
            .map(|(c, n)| {
                let name = names
                    .get_item(*c as usize)
                    .and_then(|o| o.extract::<String>())
                    .unwrap_or_else(|_| "?".into());
                if *n == 1 {
                    format!("1 {name}")
                } else {
                    format!("{n} {name}s")
                }
            })
            .collect();
        Ok(format!("Detections({})", parts.join(", ")))
    }
}

/// A loaded YOLO26 detector.
#[pyclass]
struct Detector {
    engine: Arc<dyn DetectEngine>,
    names: Py<PyList>,
}

#[pymethods]
impl Detector {
    /// `Detector(tier="n", geometry="rect", models="models")`.
    ///
    /// Weights are NOT bundled — they are AGPL and stay with whoever converted
    /// them. `models` is the manifest directory `tools/diana_convert.py` wrote.
    #[new]
    #[pyo3(signature = (tier="n", geometry="rect", models="models"))]
    fn new(py: Python<'_>, tier: &str, geometry: &str, models: &str) -> PyResult<Self> {
        let geom = match geometry {
            "rect" => ffai_diana::image::Geometry::Rect,
            "square" => ffai_diana::image::Geometry::Square,
            other => {
                return Err(PyValueError::new_err(format!(
                    "geometry must be 'rect' or 'square', got '{other}'"
                )))
            }
        };
        let engine: Arc<dyn DetectEngine> =
            Arc::new(ffai_diana::engine::Yolo26::build(tier, geom, models));
        // Touch the model now so a missing manifest fails HERE, at construction,
        // rather than on the first frame of somebody's loop.
        let names = engine.class_names().to_vec();
        if names.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "no weights for tier '{tier}' in '{models}'. Convert your own \
                 .pt with tools/diana_convert.py — FFai ships none, they are AGPL."
            )));
        }
        Ok(Detector {
            engine,
            names: PyList::new(py, &names)?.unbind(),
        })
    }

    #[getter]
    fn names(&self, py: Python<'_>) -> Py<PyList> {
        self.names.clone_ref(py)
    }

    /// Detect in an `HxWx3` (RGB), `HxWx4` (RGBA) or `HxW` (grayscale) uint8 array.
    #[pyo3(signature = (image, conf=0.25, max_det=300, from_bgr=false))]
    fn detect(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<'_, u8>,
        conf: f32,
        max_det: usize,
        from_bgr: bool,
    ) -> PyResult<Detections> {
        let buf = to_image_buffer(&image, from_bgr)?;
        let opts = DetectOptions {
            confidence: conf,
            max_detections: max_det,
            iou: None,
            classes: Vec::new(),
        };
        // Release the GIL: detection is pure Rust and takes tens of ms, so
        // holding it would serialise any caller threading over frames.
        let out = py
            .allow_threads(|| self.engine.detect(&buf, &opts))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut xyxy = Vec::with_capacity(out.detections.len() * 4);
        let mut cf = Vec::with_capacity(out.detections.len());
        let mut cls = Vec::with_capacity(out.detections.len());
        for d in &out.detections {
            xyxy.extend_from_slice(&[d.x0, d.y0, d.x1, d.y1]);
            cf.push(d.confidence);
            cls.push(d.class_id);
        }
        Ok(Detections {
            names: self.names.clone_ref(py),
            xyxy,
            conf: cf,
            cls,
        })
    }
}

/// numpy array -> `ImageBuffer`, without copying more than once.
fn to_image_buffer(a: &PyReadonlyArrayDyn<'_, u8>, from_bgr: bool) -> PyResult<ImageBuffer> {
    let shape = a.shape();
    let (h, w, c) = match shape.len() {
        2 => (shape[0], shape[1], 1usize),
        3 => (shape[0], shape[1], shape[2]),
        n => {
            return Err(PyValueError::new_err(format!(
                "expected a 2-D (HxW) or 3-D (HxWxC) uint8 array, got {n} dimensions"
            )))
        }
    };
    let format = match c {
        1 => PixelFormat::Gray8,
        3 => PixelFormat::Rgb8,
        4 => PixelFormat::Rgba8,
        other => {
            return Err(PyValueError::new_err(format!(
                "expected 1, 3 or 4 channels, got {other}"
            )))
        }
    };
    // `as_slice` requires C-contiguity; a sliced or transposed view is not, and
    // reading it as if it were would silently scramble the image.
    let data = a.as_slice().map_err(|_| {
        PyValueError::new_err(
            "array must be C-contiguous — call numpy.ascontiguousarray(a) first",
        )
    })?;
    let mut data = data.to_vec();
    if from_bgr && c >= 3 {
        for px in data.chunks_exact_mut(c) {
            px.swap(0, 2);
        }
    }
    Ok(ImageBuffer {
        width: w as u32,
        height: h as u32,
        format,
        data,
    })
}

#[pymodule]
fn ffai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detector>()?;
    m.add_class::<Detections>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
