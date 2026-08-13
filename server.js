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

app.use(express.json({ limit: "20mb" }));

app.get("/health", (_req, res) => {
    res.json({ ok: true });
});

const TEMPLATE_PATH = path.join(__dirname, "new_template.jpg");
const TARGET_W = 1116;
const TARGET_H = 722;

// LOGÍSTICA radio buttons on aligned new_template.jpg
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

// Value boxes on aligned new_template.jpg
const fieldClips = {
    sitio: [709, 117, 881, 144],
    folio: [901, 117, 1082, 144],
    fecha: [711, 176, 1082, 203],

    hora_entrada: [20, 544, 293, 580],
    hora_salida: [298, 544, 564, 576],
    id_operador: [570, 541, 770, 574],
    nombre: [774, 540, 1086, 573],
    placas_vehiculo: [23, 602, 382, 632],
    placas_caja_remolque: [386, 597, 736, 627],
    numero_marchamo: [741, 592, 1091, 623],

    elaboro: [21, 664, 290, 695],
    supervisor: [297, 661, 560, 692],
    autorizo: [568, 657, 828, 690],
    operador_firma: [834, 654, 1089, 682]
};

const FIELD_HINTS = {
    sitio: "Extract only the site code text in this box (e.g. MXCD06).",
    folio: "Extract only the folio number. Digits preferred (e.g. 0001).",
    fecha: "Extract only the handwritten date. Prefer DD/MM/YYYY or YYYY-MM-DD.",
    hora_entrada: "Extract only the entry date and time.",
    hora_salida: "Extract only the exit date and time.",
    id_operador: "Extract only the operator ID.",
    nombre: "Extract only the operator full name.",
    placas_vehiculo: "Extract only the vehicle license plate.",
    placas_caja_remolque: "Extract only the trailer/box license plate.",
    numero_marchamo: "Extract only the seal/marchamo number. Digits preferred.",
    elaboro: "Extract only the handwritten name in this signature box.",
    supervisor: "Extract only the handwritten name in this signature box.",
    autorizo: "Extract only the handwritten name in this signature box.",
    operador_firma: "Extract only the handwritten name in this signature box."
};

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function cleanOcrText(text) {
    if (text == null) {
        return "";
    }

    return String(text)
        .replace(/\r/g, "")
        .replace(/\n+/g, " ")
        .replace(/`+/g, "")
        .replace(/\s+/g, " ")
        .trim();
}

function parseInteger(text) {
    if (text == null || text === "") {
        return null;
    }

    if (typeof text === "number" && Number.isFinite(text)) {
        return Math.trunc(text);
    }

    const value = String(text).replace(/[^\d]/g, "");

    if (!value) {
        return null;
    }

    return Number.parseInt(value, 10);
}

function parseDate(text) {
    text = cleanOcrText(text);

    if (!text) {
        return null;
    }

    let match = text.match(/(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})/);

    if (match) {
        const [, y, m, d] = match;
        return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
    }

    match = text.match(/(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})/);

    if (match) {
        const [, d, m, y] = match;
        return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
    }

    return text;
}

function parseDateTime(text) {
    text = cleanOcrText(text);

    if (!text) {
        return null;
    }

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

function cleanFolio(text) {
    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    const digits = cleaned.replace(/[^\d]/g, "");
    return digits || cleaned;
}

function normalizeFirmaResult(raw) {
    const filled = Boolean(raw?.filled);
    let value = raw?.value;

    if (typeof value === "string") {
        value = cleanOcrText(value) || null;
    } else {
        value = null;
    }

    if (!filled) {
        return { filled: false, value: null };
    }

    return {
        filled: true,
        value: value || "Unknown"
    };
}

// -----------------------------------------------------------------------------
// OpenCV helpers
// -----------------------------------------------------------------------------

function loadImage(imageBuffer) {
    const img = cv.imdecode(imageBuffer);

    if (img.empty) {
        throw new Error("Unable to decode input image");
    }

    const gray =
        img.channels === 1
            ? img
            : img.cvtColor(cv.COLOR_BGR2GRAY);

    return { color: img, gray };
}

function rotateToLandscape(mat) {
    if (mat.cols >= mat.rows) {
        return mat;
    }

    // 90° clockwise for portrait phone photos of landscape forms.
    try {
        if (typeof cv.rotate === "function") {
            return cv.rotate(mat, 0); // ROTATE_90_CLOCKWISE
        }
    } catch {
        // fall through
    }

    return mat.transpose().flip(1);
}

function findHomography(templateGray, inputGray) {
    const orb = new cv.ORBDetector({
        maxFeatures: 4000
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

    if (good.length < 25) {
        throw new Error(`Not enough feature matches: ${good.length}`);
    }

    const srcPoints = good.map((m) => kp1[m.queryIdx].pt);
    const dstPoints = good.map((m) => kp2[m.trainIdx].pt);

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

function matToJpeg(mat) {
    try {
        return cv.imencode(".jpg", mat, [1, 90]); // IMWRITE_JPEG_QUALITY = 1
    } catch {
        return cv.imencode(".jpg", mat);
    }
}

function matToPng(mat) {
    return cv.imencode(".png", mat);
}

function cropMat(img, x1, y1, x2, y2) {
    const left = Math.max(0, Math.min(x1, img.cols - 1));
    const top = Math.max(0, Math.min(y1, img.rows - 1));
    const right = Math.max(left + 1, Math.min(x2, img.cols));
    const bottom = Math.max(top + 1, Math.min(y2, img.rows));

    return img.getRegion(
        new cv.Rect(left, top, right - left, bottom - top)
    );
}

async function ocrClip(mat, fieldName) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const buffer = matToPng(mat);
    const base64 = buffer.toString("base64");
    const hint =
        FIELD_HINTS[fieldName] ||
        "Extract only the text in this box.";

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 40,
        messages: [
            {
                role: "system",
                content:
                    "You are an OCR engine. Return only the extracted text. " +
                    "No labels, no quotes, no explanation. " +
                    "If empty or unreadable, return an empty string."
            },
            {
                role: "user",
                content: [
                    { type: "text", text: hint },
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

    return cleanOcrText(
        response.choices?.[0]?.message?.content || ""
    );
}

async function ocrFieldClips(alignedColor) {
    const entries = await Promise.all(
        Object.entries(fieldClips).map(
            async ([name, [x1, y1, x2, y2]]) => {
                const clip = cropMat(
                    alignedColor,
                    x1,
                    y1,
                    x2,
                    y2
                );
                const text = await ocrClip(clip, name);
                return [name, text];
            }
        )
    );

    return Object.fromEntries(entries);
}

function firmaFromText(text) {
    const value = cleanOcrText(text);

    if (!value) {
        return {
            filled: false,
            value: null
        };
    }

    return {
        filled: true,
        value
    };
}

function checkboxDarkness(gray, x, y, radius = 7) {
    let sum = 0;
    let count = 0;

    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const px = x + dx;
            const py = y + dy;

            if (
                px < 0 ||
                py < 0 ||
                px >= gray.cols ||
                py >= gray.rows
            ) {
                continue;
            }

            sum += gray.at(py, px);
            count += 1;
        }
    }

    return count ? sum / count : 255;
}

function detectSelectedOption(
    gray,
    positions,
    names,
    maxMean = 115
) {
    let best = null;

    for (let i = 0; i < positions.length; i++) {
        const [x, y] = positions[i];
        const score = checkboxDarkness(gray, x, y);

        if (score > maxMean) {
            continue;
        }

        if (!best || score < best.score) {
            best = {
                name: names[i],
                score
            };
        }
    }

    return best?.name ?? null;
}

function alignDocument(inputBuffer) {
    const templateBuffer = fs.readFileSync(TEMPLATE_PATH);
    const template = loadImage(templateBuffer);
    let { color } = loadImage(inputBuffer);

    // Phone photos are often portrait while the form is landscape.
    color = rotateToLandscape(color);

    const gray =
        color.channels === 1
            ? color
            : color.cvtColor(cv.COLOR_BGR2GRAY);

    const H = findHomography(template.gray, gray);
    const alignedColor = warpImage(color, H);
    const alignedGray =
        alignedColor.channels === 1
            ? alignedColor
            : alignedColor.cvtColor(cv.COLOR_BGR2GRAY);

    return {
        color: alignedColor,
        gray: alignedGray,
        aligned: true
    };
}

// -----------------------------------------------------------------------------
// Full-page OpenAI Vision extraction
// -----------------------------------------------------------------------------

async function extractDocumentWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 1600,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You extract data from Plastic Trade ORDEN DE SALIDA forms. " +
                    "Return JSON only with this exact shape:\n" +
                    "{\n" +
                    '  "documento": {"sitio": string|null, "folio": string|null, "fecha": "YYYY-MM-DD"|null},\n' +
                    '  "selected_logistica": string|null,\n' +
                    '  "materials": [{"material": string, "cantidad": number|null, "unidad": string|null}],\n' +
                    '  "selected_unidad": string|null,\n' +
                    '  "operador": {\n' +
                    '    "id_operador": string|null,\n' +
                    '    "nombre": string|null,\n' +
                    '    "placas_vehiculo": string|null,\n' +
                    '    "placas_caja_remolque": string|null,\n' +
                    '    "numero_marchamo": string|null\n' +
                    "  },\n" +
                    '  "horarios": {"hora_entrada": string|null, "hora_salida": string|null},\n' +
                    '  "firmas": {\n' +
                    '    "elaboro": {"filled": boolean, "value": string|null},\n' +
                    '    "supervisor": {"filled": boolean, "value": string|null},\n' +
                    '    "autorizo": {"filled": boolean, "value": string|null},\n' +
                    '    "operador": {"filled": boolean, "value": string|null}\n' +
                    "  }\n" +
                    "}\n" +
                    "Rules:\n" +
                    "- Only include materials whose checkbox is selected AND that have a numeric UNIDAD value.\n" +
                    "- Material names: Playo, Carton, RSU, Tarima, Tubo de carton, Organicos, Chatarra, Otro.\n" +
                    "- unidad is the selected UNIDAD DE MEDIDA (A granel, Pacas, Gaylord's, Barcinas, Piezas).\n" +
                    "- selected_logistica: look ONLY at which LOGÍSTICA radio circle is filled black. " +
                    "Options in order: TAGA, SERRANO, JUAN CARLOS, TREESEVER, ROSSET, PLASTIC, BIOAMBIENTALISTIK. " +
                    "Return that exact selected name. Do not guess from nearby text.\n" +
                    "- selected_unidad is the selected transport unit (e.g. Caja seca).\n" +
                    "- folio: prefer digits only (N° 0001 -> 0001).\n" +
                    "- horarios: keep date and time, prefer YYYY-MM-DD HH:mm.\n" +
                    "- firmas: filled=true if name/signature present; value=readable name or Unknown; empty box => filled=false, value=null.\n" +
                    "- Read handwriting carefully (9 vs Y, 6 vs 0, 5 vs S). Do not invent values."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Extract all filled fields from this ORDEN DE SALIDA image."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        throw new Error("OpenAI returned invalid JSON for document extraction");
    }
}

function normalizeExtractedDocument(raw, overrides = {}) {
    const materials = Array.isArray(raw?.materials)
        ? raw.materials
            .map((item) => {
                const cantidad = parseInteger(item?.cantidad);

                return {
                    material: cleanOcrText(item?.material) || null,
                    cantidad,
                    unidad: cleanOcrText(item?.unidad) || null
                };
            })
            .filter(
                (item) =>
                    item.material &&
                    item.cantidad != null &&
                    item.cantidad > 0
            )
        : [];

    const firmasRaw = raw?.firmas || {};
    const selectedLogistica =
        overrides.logistica ||
        cleanOcrText(raw?.selected_logistica) ||
        cleanOcrText(raw?.logistica) ||
        null;

    const fields = overrides.fields || {};

    return {
        documento: {
            sitio:
                cleanOcrText(fields.sitio) ||
                cleanOcrText(raw?.documento?.sitio) ||
                null,
            folio: cleanFolio(
                fields.folio || raw?.documento?.folio
            ),
            fecha: parseDate(
                fields.fecha || raw?.documento?.fecha
            )
        },
        selected_logistica: selectedLogistica,
        materials,
        selected_unidad:
            cleanOcrText(raw?.selected_unidad) || null,
        operador: {
            id_operador:
                cleanOcrText(fields.id_operador) ||
                cleanOcrText(raw?.operador?.id_operador) ||
                null,
            nombre:
                cleanOcrText(fields.nombre) ||
                cleanOcrText(raw?.operador?.nombre) ||
                null,
            placas_vehiculo:
                cleanOcrText(fields.placas_vehiculo) ||
                cleanOcrText(raw?.operador?.placas_vehiculo) ||
                null,
            placas_caja_remolque:
                cleanOcrText(fields.placas_caja_remolque) ||
                cleanOcrText(
                    raw?.operador?.placas_caja_remolque
                ) ||
                null,
            numero_marchamo:
                cleanOcrText(fields.numero_marchamo) ||
                cleanOcrText(raw?.operador?.numero_marchamo) ||
                null
        },
        horarios: {
            hora_entrada: parseDateTime(
                fields.hora_entrada ||
                    raw?.horarios?.hora_entrada
            ),
            hora_salida: parseDateTime(
                fields.hora_salida || raw?.horarios?.hora_salida
            )
        },
        firmas: {
            elaboro: fields.elaboro
                ? firmaFromText(fields.elaboro)
                : normalizeFirmaResult(firmasRaw.elaboro),
            supervisor: fields.supervisor
                ? firmaFromText(fields.supervisor)
                : normalizeFirmaResult(firmasRaw.supervisor),
            autorizo: fields.autorizo
                ? firmaFromText(fields.autorizo)
                : normalizeFirmaResult(firmasRaw.autorizo),
            operador: fields.operador_firma
                ? firmaFromText(fields.operador_firma)
                : normalizeFirmaResult(firmasRaw.operador)
        }
    };
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

    let imageForOcr;
    let alignedGray = null;
    let alignedColor = null;

    try {
        const aligned = alignDocument(inputBuffer);
        imageForOcr = aligned.color;
        alignedColor = aligned.color;
        alignedGray = aligned.gray;
        console.log("Document aligned to template");
    } catch (error) {
        console.warn(
            "Alignment failed, using orientation-normalized original:",
            error.message
        );

        let { color } = loadImage(inputBuffer);
        color = rotateToLandscape(color);
        imageForOcr = color;
    }

    // Prefer OpenCV filled-circle detection for LOGÍSTICA.
    let logisticaOverride = null;

    if (alignedGray) {
        logisticaOverride = detectSelectedOption(
            alignedGray,
            logisticas_cp,
            logisticas
        );

        if (logisticaOverride) {
            console.log(
                "LOGÍSTICA detected by checkbox:",
                logisticaOverride
            );
        }
    }

    // Crop OCR for fixed field boxes on the aligned template.
    let fieldOverride = null;

    if (alignedColor) {
        fieldOverride = await ocrFieldClips(alignedColor);
        console.log("Field crop OCR:", fieldOverride);
    }

    const jpegBuffer = matToJpeg(imageForOcr);
    const extracted = await extractDocumentWithVision(jpegBuffer);

    return normalizeExtractedDocument(extracted, {
        logistica: logisticaOverride,
        fields: fieldOverride
    });
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

        if (base64.includes(",")) {
            base64 = base64.split(",")[1];
        }

        const inputBuffer = Buffer.from(base64, "base64");

        if (!inputBuffer.length) {
            return res.status(400).json({
                error: "Invalid base64 image"
            });
        }

        const result = await processDocument(inputBuffer);
        return res.json(result);
    } catch (error) {
        console.error(error);

        return res.status(500).json({
            error: error.message
        });
    }
});

async function start() {
    if (!OPENAI_API_KEY) {
        throw new Error(
            "OPENAI_API_KEY environment variable is required"
        );
    }

    console.log(`Using OpenAI Vision model: ${OPENAI_MODEL}`);

    app.listen(PORT, () => {
        console.log(`OCR server listening on port ${PORT}`);
    });
}

async function shutdown() {
    console.log("Shutting down...");
    process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

start().catch((error) => {
    console.error("Failed to start server:", error);
    process.exit(1);
});
