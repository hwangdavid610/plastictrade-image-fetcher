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

const KILOGRAM_TEMPLATE_PATH = path.join(
    __dirname,
    "kilogram_template.png"
);
const KILOGRAM_W = 313;
const KILOGRAM_H = 733;

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
    [84, 232],
    [221, 236],
    [347, 236],
    [496, 236],
    [635, 235],
    [749, 237],
    [868, 236]
];

// UNIDAD DE TRANSPORTE radio buttons (right column)
const unidads = [
    "Caja seca",
    "Tolva 30m3",
    "Remolque",
    "Torthon",
    "Cartucho",
    "Olla 17m3",
    "Camioneta",
    "Tolva 7m3",
    "Contenedores CGR"
];

const unidads_cp = [
    [936, 293],
    [934, 318],
    [938, 344],
    [936, 370],
    [938, 395],
    [934, 421],
    [938, 447],
    [937, 475],
    [939, 499]
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

    let cleaned = String(text)
        .replace(/\r/g, "")
        .replace(/\n+/g, " ")
        .replace(/`+/g, "")
        .replace(/^["'\s]+|["'\s]+$/g, "")
        .replace(/\s+/g, " ")
        .trim();

    // Vision/OCR sometimes apologizes on blank crops instead of returning "".
    if (
        !cleaned ||
        cleaned === '""' ||
        cleaned === "''" ||
        /i'?m sorry/i.test(cleaned) ||
        /can'?t (extract|read|help)/i.test(cleaned) ||
        /unable to (extract|read|determine)/i.test(cleaned) ||
        /no text/i.test(cleaned) ||
        /blank or unreadable/i.test(cleaned) ||
        /^(n\/?a|none|null|unknown)$/i.test(cleaned)
    ) {
        return "";
    }

    return cleaned;
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

function parseWeight(text) {
    if (text == null || text === "") {
        return null;
    }

    if (typeof text === "number" && Number.isFinite(text)) {
        return text;
    }

    let cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    // Keep digits, comma, and dot. Examples: 20,060.00 / 20060.00 / 20.060,00
    cleaned = cleaned.replace(/[^\d.,]/g, "");

    if (!cleaned) {
        return null;
    }

    const hasComma = cleaned.includes(",");
    const hasDot = cleaned.includes(".");

    if (hasComma && hasDot) {
        // Assume the last separator is the decimal mark.
        if (cleaned.lastIndexOf(",") > cleaned.lastIndexOf(".")) {
            cleaned = cleaned.replace(/\./g, "").replace(",", ".");
        } else {
            cleaned = cleaned.replace(/,/g, "");
        }
    } else if (hasComma) {
        // 20060,00 or 20,060
        const parts = cleaned.split(",");

        if (parts.length === 2 && parts[1].length <= 2) {
            cleaned = `${parts[0]}.${parts[1]}`;
        } else {
            cleaned = cleaned.replace(/,/g, "");
        }
    }

    const value = Number.parseFloat(cleaned);

    return Number.isFinite(value) ? value : null;
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
    let value = raw?.value;

    if (typeof value === "string") {
        value = cleanOcrText(value) || null;
    } else {
        value = null;
    }

    const filled = Boolean(raw?.filled) && Boolean(value);

    if (!filled) {
        return { filled: false, value: null };
    }

    return {
        filled: true,
        value
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

function rotate90Clockwise(mat) {
    try {
        if (typeof cv.rotate === "function") {
            return cv.rotate(mat, 0); // ROTATE_90_CLOCKWISE
        }
    } catch {
        // fall through
    }

    return mat.transpose().flip(1);
}

function rotateToLandscape(mat) {
    if (mat.cols >= mat.rows) {
        return mat;
    }

    // 90° clockwise for portrait phone photos of landscape forms.
    return rotate90Clockwise(mat);
}

function rotateToPortrait(mat) {
    if (mat.rows >= mat.cols) {
        return mat;
    }

    // Thermal receipts are portrait; landscape photos are usually sideways.
    return rotate90Clockwise(mat);
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

function warpImage(img, H, width = TARGET_W, height = TARGET_H) {
    return img.warpPerspective(
        H,
        new cv.Size(width, height),
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

function clipLooksBlank(mat) {
    try {
        const gray =
            mat.channels === 1
                ? mat
                : mat.cvtColor(cv.COLOR_BGR2GRAY);
        const data = gray.getDataAsArray();
        let dark = 0;
        let total = 0;
        let sum = 0;

        for (const row of data) {
            for (const value of row) {
                total += 1;
                sum += value;

                if (value < 140) {
                    dark += 1;
                }
            }
        }

        if (!total) {
            return true;
        }

        // Mostly white paper with almost no ink strokes.
        return sum / total > 200 && dark / total < 0.02;
    } catch {
        return false;
    }
}

async function ocrClip(mat, fieldName) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    if (clipLooksBlank(mat)) {
        return "";
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
                    "No labels, no quotes, no explanation, no apologies. " +
                    "If the box is blank or unreadable, return an empty string and nothing else."
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

function sampleGray(gray, x, y) {
    if (
        x < 0 ||
        y < 0 ||
        x >= gray.cols ||
        y >= gray.rows
    ) {
        return null;
    }

    return gray.at(y, x);
}

// Filled radio = dark center + lighter ring. Avoids nearby text (e.g. PLASTIC).
function filledCircleScore(gray, x, y) {
    let coreSum = 0;
    let coreCount = 0;
    let ringSum = 0;
    let ringCount = 0;

    for (let dy = -8; dy <= 8; dy++) {
        for (let dx = -8; dx <= 8; dx++) {
            const dist2 = dx * dx + dy * dy;
            const value = sampleGray(gray, x + dx, y + dy);

            if (value == null) {
                continue;
            }

            if (dist2 <= 9) {
                // r <= 3
                coreSum += value;
                coreCount += 1;
            } else if (dist2 >= 25 && dist2 <= 64) {
                // 5 <= r <= 8
                ringSum += value;
                ringCount += 1;
            }
        }
    }

    if (!coreCount || !ringCount) {
        return null;
    }

    const core = coreSum / coreCount;
    const ring = ringSum / ringCount;
    const contrast = ring - core;

    // Blurry phone photos often darken the ring (~100-130), so do not
    // require a bright white ring. Require a dark core + positive contrast.
    if (core > 45 || contrast < 50) {
        return null;
    }

    return {
        core,
        ring,
        contrast
    };
}

function bestFilledNear(gray, x, y, radius = 8) {
    let best = null;

    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            const score = filledCircleScore(gray, x + dx, y + dy);

            if (!score) {
                continue;
            }

            if (!best || score.core < best.core) {
                best = {
                    ...score,
                    x: x + dx,
                    y: y + dy
                };
            }
        }
    }

    return best;
}

function detectSelectedOption(gray, positions, names) {
    let best = null;

    for (let i = 0; i < positions.length; i++) {
        const [x, y] = positions[i];
        const score = bestFilledNear(gray, x, y, 8);

        if (!score) {
            continue;
        }

        if (!best || score.core < best.core) {
            best = {
                name: names[i],
                ...score
            };
        }
    }

    return best?.name ?? null;
}

function alignDocument(
    inputBuffer,
    {
        templatePath = TEMPLATE_PATH,
        width = TARGET_W,
        height = TARGET_H,
        orientation = "landscape"
    } = {}
) {
    const templateBuffer = fs.readFileSync(templatePath);
    const template = loadImage(templateBuffer);
    let { color } = loadImage(inputBuffer);

    color =
        orientation === "portrait"
            ? rotateToPortrait(color)
            : rotateToLandscape(color);

    const gray =
        color.channels === 1
            ? color
            : color.cvtColor(cv.COLOR_BGR2GRAY);

    const H = findHomography(template.gray, gray);
    const alignedColor = warpImage(color, H, width, height);
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
                    "- selected_unidad: look ONLY at which UNIDAD DE TRANSPORTE radio circle is filled black. " +
                    "Options in order: Caja seca, Tolva 30m3, Remolque, Torthon, Cartucho, Olla 17m3, Camioneta, Tolva 7m3, Contenedores CGR. " +
                    "Return that exact selected name. Do not guess from nearby text.\n" +
                    "- folio: prefer digits only (N° 0001 -> 0001).\n" +
                    "- horarios: keep date and time, prefer YYYY-MM-DD HH:mm. Empty box => null.\n" +
                    "- operador fields: empty box => null. Never apologize or explain.\n" +
                    "- firmas: filled=true if name/signature present; value=readable name or Unknown; empty box => filled=false, value=null.\n" +
                    "- Read handwriting carefully (9 vs Y, 6 vs 0, 5 vs S). Do not invent values. Never return apology text."
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

function pickFirma(fields, firmasRaw, cropKey, ...visionKeys) {
    if (Object.prototype.hasOwnProperty.call(fields || {}, cropKey)) {
        return firmaFromText(fields[cropKey]);
    }

    for (const key of visionKeys) {
        if (firmasRaw?.[key]) {
            return normalizeFirmaResult(firmasRaw[key]);
        }
    }

    return {
        filled: false,
        value: null
    };
}

function buildResponse(raw, overrides = {}) {
    const fields = overrides.fields || {};
    const firmasRaw = raw?.firmas || {};

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

    // Strict whitelist — never pass through raw Vision / legacy keys.
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
        selected_logistica:
            overrides.logistica ||
            cleanOcrText(raw?.selected_logistica) ||
            cleanOcrText(raw?.logistica) ||
            null,
        materials,
        selected_unidad:
            overrides.unidad ||
            cleanOcrText(raw?.selected_unidad) ||
            null,
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
                    raw?.horarios?.hora_entrada ||
                    raw?.cliente_de_servicio
                        ?.fecha_hora_entrada_sitio
            ),
            hora_salida: parseDateTime(
                fields.hora_salida ||
                    raw?.horarios?.hora_salida ||
                    raw?.cliente_de_servicio
                        ?.fecha_hora_salida_sitio
            )
        },
        firmas: {
            elaboro: pickFirma(
                fields,
                firmasRaw,
                "elaboro",
                "elaboro",
                "elaboro_plastict"
            ),
            supervisor: pickFirma(
                fields,
                firmasRaw,
                "supervisor",
                "supervisor",
                "responsable_supervisor"
            ),
            autorizo: pickFirma(
                fields,
                firmasRaw,
                "autorizo",
                "autorizo",
                "autoriza_melii"
            ),
            operador: pickFirma(
                fields,
                firmasRaw,
                "operador_firma",
                "operador",
                "recibio_y_entrego_operador"
            )
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

    // Prefer OpenCV filled-circle detection for LOGÍSTICA / UNIDAD.
    let logisticaOverride = null;
    let unidadOverride = null;

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

        unidadOverride = detectSelectedOption(
            alignedGray,
            unidads_cp,
            unidads
        );

        if (unidadOverride) {
            console.log(
                "UNIDAD DE TRANSPORTE detected by checkbox:",
                unidadOverride
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

    return buildResponse(extracted, {
        logistica: logisticaOverride,
        unidad: unidadOverride,
        fields: fieldOverride
    });
}

// -----------------------------------------------------------------------------
// Kilogram receipt (NOTA DE ENTRADA) processing
// -----------------------------------------------------------------------------

async function extractKilogramWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 300,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You extract weight values from Recicladora TAGA NOTA DE ENTRADA receipts. " +
                    "Return JSON only with this exact shape:\n" +
                    "{\n" +
                    '  "bruto": number|null,\n' +
                    '  "tara": number|null,\n' +
                    '  "neto": number|null\n' +
                    "}\n" +
                    "Rules:\n" +
                    "- Read only the numeric values next to BRUTO, TARA, and NETO.\n" +
                    "- Return plain numbers (e.g. 20060 or 20060.00), not currency strings.\n" +
                    "- Ignore COSTO, TOTAL, ticket numbers, and dates.\n" +
                    "- If a value is missing or unreadable, return null.\n" +
                    "- Do not invent values. Never apologize."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Extract BRUTO, TARA, and NETO from this NOTA DE ENTRADA image."
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
        throw new Error(
            "OpenAI returned invalid JSON for kilogram extraction"
        );
    }
}

function buildKilogramResponse(raw) {
    return {
        bruto: parseWeight(raw?.bruto),
        tara: parseWeight(raw?.tara),
        neto: parseWeight(raw?.neto)
    };
}

async function processKilogramDocument(inputBuffer) {
    if (!fs.existsSync(KILOGRAM_TEMPLATE_PATH)) {
        throw new Error(
            `Template file not found: ${KILOGRAM_TEMPLATE_PATH}`
        );
    }

    let imageForOcr;

    try {
        const aligned = alignDocument(inputBuffer, {
            templatePath: KILOGRAM_TEMPLATE_PATH,
            width: KILOGRAM_W,
            height: KILOGRAM_H,
            orientation: "portrait"
        });
        imageForOcr = aligned.color;
        console.log("Kilogram receipt aligned to template");
    } catch (error) {
        console.warn(
            "Kilogram alignment failed, using orientation-normalized original:",
            error.message
        );

        let { color } = loadImage(inputBuffer);
        color = rotateToPortrait(color);
        imageForOcr = color;
    }

    const jpegBuffer = matToJpeg(imageForOcr);
    const extracted = await extractKilogramWithVision(jpegBuffer);

    return buildKilogramResponse(extracted);
}

function normalizeProcessFlag(flag) {
    const value = String(flag ?? "sheet")
        .trim()
        .toLowerCase();

    if (
        value === "kilogram" ||
        value === "kilograms" ||
        value === "kg" ||
        value === "peso"
    ) {
        return "kilogram";
    }

    if (
        !value ||
        value === "sheet" ||
        value === "orden" ||
        value === "salida"
    ) {
        return "sheet";
    }

    return null;
}

// -----------------------------------------------------------------------------
// API
// -----------------------------------------------------------------------------

app.post("/process", async (req, res) => {
    try {
        const { image, flag } = req.body;

        if (!image) {
            return res.status(400).json({
                error: "Missing 'image' field"
            });
        }

        const processFlag = normalizeProcessFlag(flag ?? "sheet");

        if (!processFlag) {
            return res.status(400).json({
                error: "Invalid 'flag'. Use 'sheet' or 'kilogram'."
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

        const result =
            processFlag === "kilogram"
                ? await processKilogramDocument(inputBuffer)
                : await processDocument(inputBuffer);

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
