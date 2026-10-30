// Dev-only worst-case data for the plate gallery (/galeria), based on the break-ui skill.
// Paste into the browser console on /galeria (or inject with a browser tool). It swaps the
// /api/plate_gallery response at the fetch boundary, so the page code under test is unchanged.
// Modes: window.__galleryMode = "demo" | "worst" | "empty" | "one", then loadGallery().
(() => {
  const realFetch = window.__realFetch || window.fetch.bind(window);
  window.__realFetch = realFetch;
  window.__galleryMode = window.__galleryMode || "worst";

  const longCam = "Pumatronix_ITSCAM450LMS_SP-280_Rodovia_Castello_Branco_km134_Sentido_Capital_Faixa3";
  const worstItems = (base) => [
    { ...base, cam_name: longCam },                                        // camera name at a realistic upper length
    { ...base, plate: "", text: "1BRASILFZW5A95MERCOSULBR", raw_text: "1BRASILFZW5A95MERCOSULBR" }, // OCR read the whole plate frame
    { ...base, plate: "", text: "", raw_text: "" },                        // detection without any OCR text
    { ...base, conf: 1, engine: "ensemble(rapidocr+easyocr+tesseract)" },  // long engine label, 100%
    { ...base, conf: 0.004, scene_url: "/missing/scene.jpg", area_url: "/missing/area.jpg" }, // deleted images
    { ...base, cam_name: "C", time: "", scene_res: "", area_res: "", plate_box: null }, // one-letter camera, fields missing
    { ...base, plate: "ÁBC1D23", cam_name: "Câmera Avenida São João — Viaduto Santa Ifigênia" }, // accents
  ];
  const worstStats = { cameras_online: 1284, cameras_total: 1300, valid_plates: 1254321, total: 9876543, today: 123456, avg_conf: 99.99, active_engine: "ensemble(rapidocr+easyocr+tesseract)" };

  window.fetch = async (url, opts) => {
    const u = String(url);
    if (!u.startsWith("/api/plate_gallery") || window.__galleryMode === "demo") return realFetch(url, opts);
    const data = await (await realFetch(url, opts)).json();
    const base = (data.items || [])[0] || { plate: "FZW5A95", text: "FZW5A95", conf: 0.75, engine: "rapidocr", cam_name: "Cam", time: "2026-10-29 17:45:06", scene_res: "1280x720", area_res: "334x169", plate_box: [470, 304, 619, 353] };
    const mode = window.__galleryMode;
    if (mode === "empty") { data.items = []; data.stats = { cameras_online: 0, cameras_total: 0, valid_plates: 0, total: 0, today: 0, avg_conf: 0 }; }
    if (mode === "one") { data.items = [base]; data.stats = { ...data.stats, cameras_online: 1, cameras_total: 1, valid_plates: 1, total: 1, today: 1 }; }
    if (mode === "worst") {
      data.items = [...worstItems(base), ...data.items.slice(0, 243)];
      data.stats = worstStats;
      data.cameras = [...(data.cameras || []), longCam];
    }
    return new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } });
  };
})();
