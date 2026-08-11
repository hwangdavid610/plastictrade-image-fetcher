const express = require("express");
const cv = require("@u4/opencv4nodejs");
const OpenAI = require("openai");
const fs = require("fs");
const path = require("path");

const app = express();

const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

if (!OPENAI_API_KEY) {
    console.warn(
        "Warning: OPENAI_API_KEY is not set. /process will fail until it is configured."
    );
}

const openai = new OpenAI({
    apiKey: OPENAI_API_KEY
});

// Increase this if your images are large.
app.use(express.json({ limit: "20mb" }));

app.get("/health", (_req, res) => {
    res.json({ ok: true });
});


const TEMPLATE_PATH = path.join(__dirname, "template.jpg");

const TARGET_W = 2100;
const TARGET_H = 1650;

// -----------------------------------------------------------------------------
// Document configuration
// -----------------------------------------------------------------------------

const materials = [
    "Playo",
    "Carton",
    "RSU",
    "Tarima",
    "Tubo de carton",
    "Organicos",
    "Chatarra"
];

const materials_cp = [
    [104, 454],
    [104, 533],
    [104, 601],
    [104, 676],
    [104, 748],
    [104, 823],
    [104, 897]
];

const udms = [
    "A granel",
    "Pacas",
    "Gaylord's",
    "Barcinas",
    "A granel",
    "Gaylord's",
    "A granel",
    "Piezas",
    "Piezas",
    "A granel",
    "A granel",
    "Agranel",
    "Pacas",
    "Gaylord's",
    "Barcinas"
];

const udms_cp = [
    [755, 452],
    [896, 452],
    [1013, 452],
    [1166, 452],
    [845, 525],
    [1035, 525],
    [951, 599],
    [960, 674],
    [960, 747],
    [951, 817],
    [951, 891],
    [755, 966],
    [896, 966],
    [1011, 966],
    [1162, 966]
];

const cantidad_pos = [549, 496, 720, 558];

const unidads = [
    "Caja seca",
    "Tolva 30m3",
    "Remolque",
    "Torthon",
    "Cartucho",
    "Olla 17m3",
    "Camioneta",
    "Tolva 7m3",
    "Contendores CGR"
];

const unidads_cp = [
    [1329, 460],
    [1329, 524],
    [1329, 586],
    [1328, 648],
    [1328, 710],
    [1328, 772],
    [1327, 834],
    [1327, 896],
    [1327, 961]
];

const kg_pos = [1637, 419, 1943, 484];

const clips = [
    ["sitio", [1277, 185, 1615, 283]],
    ["folio", [1635, 185, 1993, 280]],
    ["fecha", [1434, 302, 1995, 359]],

    ["oper1", [71, 1054, 490, 1151]],
    ["oper2", [496, 1054, 726, 1151]],
    ["oper3", [737, 1054, 1288, 1151]],
    ["oper4", [1628, 1054, 1990, 1151]],

    ["time11", [515, 1201, 1025, 1250]],
    ["time12", [515, 1250, 1025, 1312]],
    ["time21", [1526, 1201, 1990, 1250]],
    ["time22", [1526, 1250, 1990, 1312]]
];

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function cleanOcrText(text) {
    return text
        .replace(/\r/g, "")
        .replace(/\n+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function parseInteger(text) {
    if (!text) {
        return null;
    }

    // Keep digits only.
    const value = text.replace(/[^\d]/g, "");

    if (!value) {
        return null;
    }

    return Number.parseInt(value, 10);
}

function parseDate(text) {
    text = cleanOcrText(text);

    // Try YYYY-MM-DD
    let match = text.match(
        /(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})/
    );

    if (match) {
        const [, y, m, d] = match;

        return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
    }

    // Try DD/MM/YYYY
    match = text.match(
        /(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})/
    );

    if (match) {
        const [, d, m, y] = match;

        return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
    }

    return text;
}

function parseDateTime(text) {
    text = cleanOcrText(text);

    // Extract date and time.
    const match = text.match(
        /(\d{1,2})[-\/](\d{1,2})[-\/](\d{2,4}).*?(\d{1,2}):(\d{2})/
    );

    if (!match) {
        return text;
    }

    let [, d, m, y, hour, minute] = match;

    if (y.length === 2) {
        y = `20${y}`;
    }

    return (
        `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")} ` +
        `${hour.padStart(2, "0")}:${minute}`
    );
}

// -----------------------------------------------------------------------------
// OpenCV processing
// -----------------------------------------------------------------------------

function loadImage(imageBuffer) {
    const img = cv.imdecode(imageBuffer);

    if (img.empty) {
        throw new Error("Unable to decode input image");
    }

    const gray = img.cvtColor(cv.COLOR_BGR2GRAY);

    return {
        color: img,
        gray
    };
}

function findHomography(templateGray, inputGray) {
    // @u4/opencv4nodejs exposes detect() + compute(), not detectAndCompute().
    const orb = new cv.ORBDetector({
        maxFeatures: 3000
    });

    const kp1 = orb.detect(templateGray);
    const des1 = orb.compute(templateGray, kp1);

    const kp2 = orb.detect(inputGray);
    const des2 = orb.compute(inputGray, kp2);

    if (!des1 || !des2 || des1.rows === 0 || des2.rows === 0) {
        throw new Error("No features found");
    }

    const matches = cv.matchKnnBruteForceHamming(des1, des2, 2);

    const good = [];

    for (const pair of matches) {
        if (pair.length < 2) {
            continue;
        }

        const [m, n] = pair;

        if (m.distance < 0.75 * n.distance) {
            good.push(m);
        }
    }

    if (good.length < 20) {
        throw new Error(
            `Not enough feature matches: ${good.length}`
        );
    }

    const srcPoints = good.map(
        m => kp1[m.queryIdx].pt
    );

    const dstPoints = good.map(
        m => kp2[m.trainIdx].pt
    );

    // Same direction as the original Python:
    //
    // H = findHomography(dst, src, RANSAC, 5)
    //
    const result = cv.findHomography(
        dstPoints,
        srcPoints,
        cv.RANSAC,
        5
    );

    if (!result || !result.homography) {
        throw new Error("Unable to calculate homography");
    }

    return result.homography;
}

function warpImage(img, H) {
    return img.warpPerspective(
        H,
        new cv.Size(TARGET_W, TARGET_H),
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Vec3(255, 255, 255)
    );
}

function convertBW(img) {
    const result = img.threshold(
        150,
        255,
        cv.THRESH_BINARY
    );

    return result;
}

function findCP(img, positions, names) {
    const result = [];

    for (let i = 0; i < positions.length; i++) {
        const [x, y] = positions[i];

        const value = img.at(y, x);

        if (value === 0) {
            result.push({
                y,
                name: names[i]
            });
        }
    }

    return result;
}

function cropMat(img, x1, y1, x2, y2) {
    return img.getRegion(
        new cv.Rect(
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )
    );
}

function matToPng(mat) {
    return cv.imencode(".png", mat);
}

// -----------------------------------------------------------------------------
// OCR (OpenAI Vision)
// -----------------------------------------------------------------------------

const FIELD_HINTS = {
    sitio: "Extract the site/location name written in this box.",
    folio: "Extract the folio/document number. Digits only if possible.",
    fecha: "Extract the date. Prefer YYYY-MM-DD or DD/MM/YYYY.",
    oper1: "Extract the operator full name.",
    oper2: "Extract the operator ID.",
    oper3: "Extract the vehicle license plate.",
    oper4: "Extract the trailer/box license plate.",
    time11: "Extract the date and time of site entry.",
    time12: "Extract the date and time of site exit.",
    time21: "Extract the date and time of warehouse entry.",
    time22: "Extract the date and time of warehouse exit.",
    cantidad: "Extract the quantity as an integer number only.",
    kg: "Extract the kilograms as an integer number only."
};

async function ocrMat(mat, fieldName = "text") {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const buffer = matToPng(mat);
    const base64 = buffer.toString("base64");
    const hint = FIELD_HINTS[fieldName] ||
        "Extract the handwritten or printed text in this image.";

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 80,
        messages: [
            {
                role: "system",
                content:
                    "You are an OCR engine for Spanish industrial forms. " +
                    "Return only the extracted text. No labels, no quotes, no explanation. " +
                    "If the field is empty or unreadable, return an empty string."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: hint
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/png;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "";

    return cleanOcrText(text);
}

// -----------------------------------------------------------------------------
// Main document processing
// -----------------------------------------------------------------------------

async function processDocument(inputBuffer) {
    if (!fs.existsSync(TEMPLATE_PATH)) {
        throw new Error(
            `Template file not found: ${TEMPLATE_PATH}`
        );
    }

    const templateBuffer = fs.readFileSync(TEMPLATE_PATH);

    const template = loadImage(templateBuffer);
    const input = loadImage(inputBuffer);

    // Same processing sequence as proc.py:
    //
    // 1. ORB
    // 2. BFMatcher
    // 3. RANSAC homography
    // 4. Warp
    // 5. Binary threshold (checkboxes only)
    //
    const H = findHomography(
        template.gray,
        input.gray
    );

    const aligned = warpImage(
        input.gray,
        H
    );

    const bw = convertBW(aligned);

    // -------------------------------------------------------------------------
    // Checkbox detection
    // -------------------------------------------------------------------------

    const selectedMaterials = findCP(
        bw,
        materials_cp,
        materials
    );

    const selectedUdms = findCP(
        bw,
        udms_cp,
        udms
    );

    const selectedUnidad = findCP(
        bw,
        unidads_cp,
        unidads
    );

    // -------------------------------------------------------------------------
    // OCR clips (grayscale is better for Vision than binary)
    // -------------------------------------------------------------------------

    const ocrEntries = await Promise.all(
        clips.map(async ([name, [x1, y1, x2, y2]]) => {
            const clip = cropMat(
                aligned,
                x1,
                y1,
                x2,
                y2
            );

            const text = await ocrMat(clip, name);

            return [name, text];
        })
    );

    const ocr = Object.fromEntries(ocrEntries);

    // -------------------------------------------------------------------------
    // OCR quantity / kilogram fields
    // -------------------------------------------------------------------------

    const quantities = await Promise.all(
        selectedMaterials.map(async (material) => {
            const y = material.y;

            const cantidad = cropMat(
                aligned,
                cantidad_pos[0],
                y - 34,
                cantidad_pos[2],
                y + 34
            );

            const cantidadText = await ocrMat(
                cantidad,
                "cantidad"
            );

            return parseInteger(cantidadText);
        })
    );

    const kilogramResults = await Promise.all(
        Array.from({ length: 8 }, async (_, i) => {
            const y = Math.floor(415 + 72.5 * (0.5 + i));

            const kg = cropMat(
                aligned,
                kg_pos[0],
                y - 34,
                kg_pos[2],
                y + 34
            );

            const kgText = await ocrMat(kg, "kg");

            if (!kgText) {
                return null;
            }

            return parseInteger(kgText);
        })
    );

    const kilograms = kilogramResults.filter(
        (value) => value !== null
    );

    // -------------------------------------------------------------------------
    // Construct result matching schema.json
    // -------------------------------------------------------------------------

    const result = {
        documento: {
            sitio: ocr.sitio,
            folio: ocr.folio,
            fecha: parseDate(ocr.fecha)
        },

        materials: selectedMaterials.map(
            (material, index) => ({
                material: material.name,
                cantidad: quantities[index],
                unidad:
                    selectedUdms[index]?.name ?? null
            })
        ),

        selected_unidad:
            selectedUnidad[0]?.name ?? null,

        kilogramos: kilograms,

        operador: {
            nombre: ocr.oper1,
            id_operador: ocr.oper2,
            placas_vehiculo: ocr.oper3,
            placas_caja_remolque: ocr.oper4,

            // There is no separate numero_marchamo clip
            // in proc.py, so it cannot be reliably extracted
            // from the supplied processing definition.
            numero_marchamo: null
        },

        cliente_de_servicio: {
            fecha_hora_entrada_sitio:
                parseDateTime(ocr.time11),

            fecha_hora_salida_sitio:
                parseDateTime(ocr.time12)
        },

        almacen_de_descarga: {
            fecha_hora_entrada_almacen:
                parseDateTime(ocr.time21),

            fecha_hora_salida_almacen:
                parseDateTime(ocr.time22)
        },

        firmas: {
            elaboro_plastict: {
                valor: "firma manuscrita",
                nombre_probable: null
            },

            responsable_supervisor: {
                valor: "firma manuscrita"
            },

            autoriza_melii: {
                nombre: null
            },

            recibio_y_entrego_operador: {
                valor: "firma manuscrita"
            },

            recibio_cliente: {
                valor: "firma manuscrita"
            }
        }
    };

    return result;
}

// -----------------------------------------------------------------------------
// API
// -----------------------------------------------------------------------------

app.post("/process", async (req, res) => {
    try {
        const { image } = req.body;

        if (!image) {
            return res.status(400).json({
                error: "Missing 'image' field"
            });
        }

        let base64 = image;

        // Also support:
        // data:image/jpeg;base64,...
        if (base64.includes(",")) {
            base64 = base64.split(",")[1];
        }

        const inputBuffer = Buffer.from(
            base64,
            "base64"
        );

        if (!inputBuffer.length) {
            return res.status(400).json({
                error: "Invalid base64 image"
            });
        }

        const result = await processDocument(
            inputBuffer
        );

        return res.json(result);

    } catch (error) {
        console.error(error);

        return res.status(500).json({
            error: error.message
        });
    }
});

// -----------------------------------------------------------------------------
// Start
// -----------------------------------------------------------------------------

async function start() {
    if (!OPENAI_API_KEY) {
        throw new Error(
            "OPENAI_API_KEY environment variable is required"
        );
    }

    console.log(
        `Using OpenAI Vision model: ${OPENAI_MODEL}`
    );

    app.listen(PORT, () => {
        console.log(
            `OCR server listening on port ${PORT}`
        );
    });
}

async function shutdown() {
    console.log("Shutting down...");
    process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

start().catch(error => {
    console.error(
        "Failed to start server:",
        error
    );

    process.exit(1);
});
