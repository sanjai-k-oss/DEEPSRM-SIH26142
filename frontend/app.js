const API="http://127.0.0.1:8000";

const steps=[
["Scene validation","Checking the selected Sentinel-2 request"],
["Sentinel-2 retrieval","Requesting L2A bands from Copernicus"],
["Cloud masking","Masking cloud / cloud-shadow classes"],
["Radiometric preparation","Preparing reflectance values"],
["Patch preparation","Preparing model-ready raster"],
["Super resolution","EDSR x4 neural RGB enhancement"],
["Geo-referencing","Preserving CRS and spatial transform"],
["GeoTIFF export","Preparing GIS-ready output"]
];

const pipeline=document.getElementById("pipeline");
if(pipeline){
  steps.forEach((s,i)=>{
    const el=document.createElement("div");
    el.className="pipeline-step";
    el.id="step-"+i;
    el.innerHTML=`<span>${String(i+1).padStart(2,"0")} · ${s[0]}</span><small>${s[1]}</small>`;
    pipeline.appendChild(el);
  });
}

function sleep(ms){return new Promise(r=>setTimeout(r,ms))}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function dateISO(d){return d.toISOString().slice(0,10)}
function initDates(){
  const end=new Date(), start=new Date();
  start.setDate(end.getDate()-30);
  const a=document.getElementById("startDate"), b=document.getElementById("endDate");
  if(a)a.value=dateISO(start); if(b)b.value=dateISO(end);
}
initDates();

async function checkAPI(){
  const badge=document.getElementById("apiBadge");
  try{
    const r=await fetch(API+"/api/status");
    if(r.ok) badge.textContent="● API online";
    else throw Error();
  }catch{badge.textContent="● API offline"}
}
checkAPI();

async function runPipeline(){
  const state=document.getElementById("pipelineState");
  if(!state)return;
  state.textContent="PROCESSING";
  for(let i=0;i<steps.length;i++){
    const el=document.getElementById("step-"+i);
    if(el)el.classList.add("active");
    await sleep(260);
    if(el){el.classList.remove("active");el.classList.add("done")}
  }
}

function showSentinelResult(data){
  const panel=document.getElementById("resultPanel");
  panel.classList.remove("hidden");
  document.getElementById("originalImg").src=API+(data.original_preview || data.preview);
  document.getElementById("enhancedImg").src=API+data.preview;
  const dl=document.getElementById("downloadBtn");
  dl.href=API+data.download;
  dl.textContent="Download Multispectral GeoTIFF";
  const aiDl=document.getElementById("aiDownloadBtn");
  if(aiDl && data.ai_rgb_download){ aiDl.href=API+data.ai_rgb_download; aiDl.classList.remove("hidden"); }
  document.querySelector(".result-head h2").textContent="Sentinel-2 DEEPSRM output";
  document.querySelector(".notice").textContent=
    "AI demo: the enhanced preview and separate RGB GeoTIFF use a pretrained EDSR ×4 neural super-resolution model. " +
    "The downloadable multispectral GeoTIFF preserves the Sentinel-2 bands using the geospatial baseline. A satellite-trained multispectral DEEPSRM checkpoint is still the final research step.";
  panel.scrollIntoView({behavior:"smooth"});
}

document.getElementById("searchBtn")?.addEventListener("click",async()=>{
  const out=document.getElementById("sentinelResults");
  out.innerHTML='<div class="empty">Searching the live Copernicus STAC catalogue…</div>';
  const body={
    latitude:+document.getElementById("lat").value,
    longitude:+document.getElementById("lon").value,
    start_date:document.getElementById("startDate").value,
    end_date:document.getElementById("endDate").value,
    max_cloud:+document.getElementById("cloud").value,
    limit:12
  };
  try{
    const r=await fetch(API+"/api/sentinel/search",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)
    });
    const data=await r.json();
    if(!r.ok)throw Error(data.detail||"Search failed");
    if(!data.features?.length){
      out.innerHTML='<div class="empty">No matching Sentinel-2 L2A scenes. Try a wider date range or higher cloud limit.</div>';
      return;
    }
    out.innerHTML=data.features.map((f,i)=>`
      <article class="scene">
        ${f.thumbnail?`<img src="${escapeHtml(f.thumbnail)}" alt="Sentinel-2 scene">`:`<div style="height:150px;background:#dce2e2"></div>`}
        <div class="scene-body">
          <strong>${escapeHtml(f.id||"Sentinel-2 L2A")}</strong>
          <p>${escapeHtml(f.datetime||"Unknown date")}<br>Cloud cover: ${f.cloud_cover ?? "N/A"}%</p>
          <button class="btn" onclick='processSelected(${JSON.stringify(body)})'>Use this scene / process</button>
        </div>
      </article>`).join("");
  }catch(e){
    out.innerHTML=`<div class="empty">${escapeHtml(e.message)}</div>`;
  }
});

window.processSelected=async(body)=>{
  document.getElementById("process")?.scrollIntoView({behavior:"smooth"});
  const state=document.getElementById("pipelineState");
  state.textContent="STARTING";
  try{
    await runPipeline();
    const r=await fetch(API+"/api/sentinel/process",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...body,size:256})
    });
    const data=await r.json();
    if(!r.ok)throw Error(data.detail||"Processing failed");
    showSentinelResult(data);
    state.textContent="COMPLETE";
  }catch(e){
    state.textContent="ERROR";
    alert("Sentinel processing failed. Make sure your CDSE credentials are configured in backend/.env and FastAPI is running.\\n\\n"+e.message);
  }
};

// Local GeoTIFF upload: real geospatial 4× baseline, not neural inference.
const fileInput=document.getElementById("fileInput");
const dropzone=document.getElementById("dropzone");
const chooseBtn=document.getElementById("chooseBtn");
const fileInfo=document.getElementById("fileInfo");
const processBtn=document.getElementById("processBtn");
let selectedFile=null;
chooseBtn?.addEventListener("click",()=>fileInput.click());
fileInput?.addEventListener("change",()=>{if(fileInput.files[0])selectFile(fileInput.files[0])});
dropzone?.addEventListener("dragover",e=>{e.preventDefault()});
dropzone?.addEventListener("drop",e=>{e.preventDefault();if(e.dataTransfer.files[0])selectFile(e.dataTransfer.files[0])});
function selectFile(file){
  selectedFile=file;
  fileInfo.classList.remove("hidden");
  fileInfo.innerHTML=`<b>${escapeHtml(file.name)}</b><br>${(file.size/1048576).toFixed(2)} MB`;
  processBtn.classList.remove("hidden");
}
processBtn?.addEventListener("click",async()=>{
  if(!selectedFile)return;
  try{
    await runPipeline();
    const fd=new FormData(); fd.append("file",selectedFile);
    const r=await fetch(API+"/api/upload/process",{method:"POST",body:fd});
    const data=await r.json();
    if(!r.ok)throw Error(data.detail||"Upload failed");
    showSentinelResult(data);
    document.getElementById("pipelineState").textContent="COMPLETE";
  }catch(e){alert(e.message)}
});
