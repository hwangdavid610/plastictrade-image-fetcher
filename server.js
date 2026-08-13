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


const TEMPLATE_PATH = path.join(__dirname, "new_template.jpg");

// Native size of new_template.jpg
const TARGET_W = 1116;
const TARGET_H = 722;

// -----------------------------------------------------------------------------
// Document configuration (new_template.jpg)
// -----------------------------------------------------------------------------

const logisticas = [
    "TAGA",
    "SERRANO",
    "JUAN CARLOS",
    "TREESEVER",
    "ROSSET",
    "PLASTIC",
    "BIOAMBIENTALISTIK"
];

const logisticas_cp = [
    [83, 236],
    [216, 236],
    [349, 236],
    [482, 236],
    [615, 236],
    [748, 236],
    [859, 236]
];

const materials = [
    "Playo",
    "Carton",
    "RSU",
    "Tarima",
    "Tubo de carton",
    "Organicos",
    "Chatarra",
    "Otro"
];

const materials_cp = [
    [46, 289],
    [46, 320],
    [46, 351],
    [46, 382],
    [46, 413],
    [46, 444],
    [46, 475],
    [46, 506]
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
    "A granel",
    "Pacas",
    "Gaylord's",
    "Barcinas"
];

const udms_cp = [
    [530, 289],
    [613, 289],
    [683, 289],
    [755, 289],
    [592, 320],
    [704, 320],
    [654, 351],
    [649, 382],
    [649, 413],
    [663, 444],
    [663, 475],
    [530, 506],
    [571, 506],
    [663, 506],
    [755, 506]
];

// UNIDAD number column on each material row.
const cantidad_pos = [300, 0, 410, 0];

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
    [754, 265],
    [754, 289],
    [754, 313],
    [754, 337],
    [754, 361],
    [754, 385],
    [754, 409],
    [754, 433],
    [754, 457]
];

const clips = [
    ["sitio", [720, 70, 900, 115]],
    ["folio", [910, 70, 1095, 115]],
    ["fecha", [910, 140, 1095, 195]],

    ["hora_entrada", [30, 548, 280, 585]],
    ["hora_salida", [290, 548, 540, 585]],
    ["id_operador", [550, 548, 800, 585]],
    ["nombre", [810, 548, 1095, 585]],

    ["placas_vehiculo", [30, 615, 370, 655]],
    ["placas_caja_remolque", [380, 615, 740, 655]],
    ["numero_marchamo", [750, 615, 1095, 655]]
];

const firmaClips = [
    ["elaboro", [30, 670, 290, 715]],
    ["supervisor", [300, 670, 560, 715]],
    ["autorizo", [570, 670, 830, 715]],
    ["operador", [840, 670, 1095, 715]]
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
    sitio: "Extract the site/location code in this box.",
    folio: "Extract the folio/document number. Digits preferred.",
    fecha: "Extract the date. Prefer YYYY-MM-DD or DD/MM/YYYY.",
    hora_entrada: "Extract the entry date and time.",
    hora_salida: "Extract the exit date and time.",
    id_operador: "Extract the operator ID.",
    nombre: "Extract the operator full name.",
    placas_vehiculo: "Extract the vehicle license plate.",
    placas_caja_remolque: "Extract the trailer/box license plate.",
    numero_marchamo: "Extract the seal/marchamo number. Digits preferred.",
    cantidad: "Extract the quantity/unidad number as an integer only."
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

function normalizeFirmaResult(raw) {
    let filled = Boolean(raw?.filled);
    let value = raw?.value;

    if (typeof value === "string") {
        value = cleanOcrText(value);
    } else {
        value = null;
    }

    if (!filled) {
        return {
            filled: false,
            value: null
        };
    }

    if (!value) {
        value = "Unknown";
    }

    return {
        filled: true,
        value
    };
}

async function ocrFirma(mat) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const buffer = matToPng(mat);
    const base64 = buffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 80,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You analyze signature boxes on Spanish industrial forms. " +
                    "Return JSON only with keys filled and value. " +
                    "filled=true if there is any handwritten signature, mark, or name. " +
                    "filled=false if the box is blank. " +
                    "value is the readable person name when possible. " +
                    "If filled but the name cannot be read, value must be \"Unknown\". " +
                    "If not filled, value must be null."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text:
                            "Inspect this signature box and return " +
                            '{"filled":boolean,"value":string|null}.'
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

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return normalizeFirmaResult(JSON.parse(text));
    } catch {
        return {
            filled: false,
            value: null
        };
    }
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

    const selectedLogistica = findCP(
        bw,
        logisticas_cp,
        logisticas
    );

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
    // OCR signature boxes
    // -------------------------------------------------------------------------

    const firmaEntries = await Promise.all(
        firmaClips.map(async ([name, [x1, y1, x2, y2]]) => {
            const clip = cropMat(
                aligned,
                x1,
                y1,
                x2,
                y2
            );

            const firma = await ocrFirma(clip);

            return [name, firma];
        })
    );

    const firmas = Object.fromEntries(firmaEntries);

    // -------------------------------------------------------------------------
    // OCR UNIDAD number for each selected material row
    // -------------------------------------------------------------------------

    const materialFields = await Promise.all(
        selectedMaterials.map(async (material) => {
            const y = material.y;

            const cantidad = cropMat(
                aligned,
                cantidad_pos[0],
                y - 16,
                cantidad_pos[2],
                y + 16
            );

            const cantidadText = await ocrMat(
                cantidad,
                "cantidad"
            );

            return {
                cantidad: parseInteger(cantidadText)
            };
        })
    );

    // -------------------------------------------------------------------------
    // Construct result
    // -------------------------------------------------------------------------

    // Only selected materials with a detected cantidad.
    const materialsResult = selectedMaterials
        .map((material, index) => {
            const udm = selectedUdms.find(
                (item) => Math.abs(item.y - material.y) <= 20
            );

            return {
                material: material.name,
                cantidad: materialFields[index].cantidad,
                unidad: udm?.name ?? null,
                kilogramos: materialFields[index].cantidad
            };
        })
        .filter((item) => item.cantidad !== null);

    const result = {
        documento: {
            sitio: ocr.sitio,
            folio: ocr.folio,
            fecha: parseDate(ocr.fecha)
        },

        logistica: selectedLogistica[0]?.name ?? null,

        materials: materialsResult,

        selected_unidad:
            selectedUnidad[0]?.name ?? null,

        operador: {
            id_operador: ocr.id_operador,
            nombre: ocr.nombre,
            placas_vehiculo: ocr.placas_vehiculo,
            placas_caja_remolque: ocr.placas_caja_remolque,
            numero_marchamo: ocr.numero_marchamo
        },

        horarios: {
            hora_entrada: parseDateTime(ocr.hora_entrada),
            hora_salida: parseDateTime(ocr.hora_salida)
        },

        firmas
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
