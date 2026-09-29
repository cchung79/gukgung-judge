/* 동작 모드: 선수 뒷모습 화면에서 만작→발시를 자세(YOLOv8n-pose)와 프레임 차이로 잡고,
   발시음·뻐꾸기(A형=우리 관) 소리와 합쳐 gukgung-judge 의 발시/관중 흐름에 넘긴다. */
(()=>{
'use strict';
const MW=640,MH=352,DS=4,HOLD_MIN=1.5,GAP_TOL=0.3,WIN=2.2,FRAC=0.35,DIFF_MIN=10;
const ORT=new URL(new URLSearchParams(location.search).get('ort')||'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',location.href).href;
const MODEL=new URL('../gwan-motion/models/yolov8n-pose_352x640.onnx',location.href).href;
const LOCK=6,SND_MATCH=0.7, RESOLVE_LAG=1.2, TONE_POST=0.8, TONE_WAIT=1.35;
const $=id=>document.getElementById(id);
const G=()=>window.__gg;
const M=window.GM={on:false,slots:[],snd:[],pend:[],tones:[],sess:null,loading:false,busy:false,lastPoseT:-9,inferMs:[],prev:null,ring:[],ringRate:16000};
const lc=document.createElement('canvas');lc.width=MW;lc.height=MH;const lx=lc.getContext('2d',{willReadFrequently:true});
const dc=document.createElement('canvas');const dx=dc.getContext('2d',{willReadFrequently:true});
const st=t=>{const e=$('gmStat');if(e)e.textContent=t};
const V=()=>G().vid();
function loadScript(u){return new Promise((res,rej)=>{const s=document.createElement('script');s.src=u;s.onload=res;s.onerror=()=>rej(new Error('스크립트 로드 실패'));document.head.appendChild(s)})}
M.init=async()=>{
  if(M.sess||M.loading)return;M.loading=true;st('자세 모델 불러오는 중… (처음 한 번 13MB)');
  try{await loadScript(ORT+'ort.wasm.min.js');ort.env.wasm.wasmPaths=ORT;ort.env.wasm.proxy=!new URLSearchParams(location.search).has('noproxy');
    ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||2):1;
    M.sess=await ort.InferenceSession.create(MODEL,{executionProviders:['wasm']});st('자세 모델 준비됨. 선수들이 서 있는 화면에서 「선수 위치 잡기」를 누르세요.');
  }catch(e){st('모델 로드 실패: '+(e&&e.message||e)+' · 주소 끝에 ?noproxy=1 을 붙여 다시 열어 보세요')}
  M.loading=false;G().updateGo();
};
/* ---------- pose ---------- */
function letterbox(){const v=V(),vw=v.videoWidth,vh=v.videoHeight,sc=Math.min(MW/vw,MH/vh),nw=Math.round(vw*sc),nh=Math.round(vh*sc),dw=(MW-nw)/2,dh=(MH-nh)/2;
  lx.fillStyle='#727272';lx.fillRect(0,0,MW,MH);lx.drawImage(v,Math.round(dw-0.1),Math.round(dh-0.1),nw,nh);return{sc,dw,dh}}
function iou(a,b){const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]),x2=Math.min(a[2],b[2]),y2=Math.min(a[3],b[3]);const i=Math.max(0,x2-x1)*Math.max(0,y2-y1);return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)}
async function pose(){
  const g=letterbox(),id=lx.getImageData(0,0,MW,MH).data,N=MW*MH,f=new Float32Array(3*N);
  for(let i=0,p=0;i<N;i++,p+=4){f[i]=id[p]/255;f[N+i]=id[p+1]/255;f[2*N+i]=id[p+2]/255}
  const t0=performance.now();
  const out=await M.sess.run({[M.sess.inputNames[0]]:new ort.Tensor('float32',f,[1,3,MH,MW])});
  M.inferMs.push(performance.now()-t0);if(M.inferMs.length>20)M.inferMs.shift();
  const o=out[M.sess.outputNames[0]],A=o.data,n=o.dims[2],c=[];
  for(let i=0;i<n;i++){const sc=A[4*n+i];if(sc>0.25){const cx=A[i],cy=A[n+i],w=A[2*n+i],h=A[3*n+i];c.push({s:sc,i,b:[cx-w/2,cy-h/2,cx+w/2,cy+h/2]})}}
  c.sort((a,b)=>b.s-a.s);const keep=[];for(const d of c)if(keep.every(k=>iou(k.b,d.b)<0.5))keep.push(d);
  return keep.map(d=>{const kp=[];for(let k=0;k<17;k++)kp.push([(A[(5+3*k)*n+d.i]-g.dw)/g.sc,(A[(6+3*k)*n+d.i]-g.dh)/g.sc,A[(7+3*k)*n+d.i]]);
    return{box:[(d.b[0]-g.dw)/g.sc,(d.b[1]-g.dh)/g.sc,(d.b[2]-g.dw)/g.sc,(d.b[3]-g.dh)/g.sc],kp,s:d.s}});
}
const upOf=p=>{const h=p.box[3]-p.box[1],sh=(p.kp[5][1]+p.kp[6][1])/2;return Math.max((sh-p.kp[9][1])/h,(sh-p.kp[10][1])/h)};
/* ---------- calibration ---------- */
M.calibrate=async()=>{
  if(!M.sess){st('모델이 아직 준비되지 않았습니다');return}
  const v=V();if(!v.videoWidth){st('영상이 아직 없습니다');return}
  $('gmCal').disabled=true;st('선수 위치 찾는 중…');
  try{
    while(M.busy)await new Promise(r=>setTimeout(r,50));M.busy=true;let ps0;try{ps0=await pose()}finally{M.busy=false}
    const ps=ps0.sort((a,b)=>(a.box[0]+a.box[2])-(b.box[0]+b.box[2]));
    M.slots=ps.map((p,i)=>({no:i+1,box:p.box,cx:(p.box[0]+p.box[2])/2,diffs:[],holdStart:null,lastPos:null,state:'대기',holdDur:0}));
    M.spacing=M.slots.length>1?(M.slots[M.slots.length-1].cx-M.slots[0].cx)/(M.slots.length-1):400;
    if(M.slots.length){$('nShooter').value=M.slots.length;$('nShooter').dispatchEvent(new Event('input'))}
    st(M.slots.length?`선수 ${M.slots.length}명 설정 (맨 왼쪽이 1번). 서 있는 사람이 다르면 다시 누르세요.`:'사람이 잡히지 않았습니다');
    setupDiff();
  }catch(e){st('위치 잡기 실패: '+(e&&e.message||e))}
  $('gmCal').disabled=false;G().updateGo();
};
/* ---------- frame diff ---------- */
function setupDiff(){const v=V();dc.width=Math.round(v.videoWidth/DS);dc.height=Math.round(v.videoHeight/DS);M.prev=null}
function diffFrame(t){
  const W=dc.width,H=dc.height;if(!W)return;dx.drawImage(V(),0,0,W,H);const d=dx.getImageData(0,0,W,H).data;
  const g=new Uint8Array(W*H);for(let i=0,p=0;i<g.length;i++,p+=4)g[i]=(d[p]*77+d[p+1]*150+d[p+2]*29)>>8;
  if(M.prev){
    const hist=new Uint32Array(256),ab=new Uint8Array(W*H);
    for(let i=0;i<g.length;i++){const a=Math.abs(g[i]-M.prev[i]);ab[i]=a;hist[a]++}
    let acc=0,med=0;for(let k=0;k<256;k++){acc+=hist[k];if(acc>=g.length/2){med=k;break}}
    for(const s of M.slots){
      const[bx0,by0,bx1,by1]=s.box,x0=Math.max(0,Math.floor(bx0/DS)-4),x1=Math.min(W,Math.ceil(bx1/DS)+4),y0=Math.max(0,Math.floor(by0/DS)),y1=Math.min(H,Math.floor((by0+(by1-by0)*0.6)/DS));
      const h2=new Uint32Array(256);let cnt=0;for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){h2[ab[y*W+x]]++;cnt++}
      let a2=0,p97=0;const target=cnt*0.97;for(let k=0;k<256;k++){a2+=h2[k];if(a2>=target){p97=k;break}}
      s.diffs.push([t,p97-med]);if(s.diffs.length>600)s.diffs.shift();
    }
  }
  M.prev=g;
}
/* ---------- 만작 → 발시 ---------- */
async function inferAt(t){M.busy=true;try{onPose(t,await pose())}catch(e){st('자세 인식 오류: '+(e&&e.message||e))}finally{M.busy=false}}
function onPose(t,ps){
  const inter=0.25,tol=Math.max(GAP_TOL,inter*1.5);
  for(const s of M.slots){
    let best=null,bd=1e9;for(const p of ps){const d=Math.abs((p.box[0]+p.box[2])/2-s.cx);if(d<bd){bd=d;best=p}}
    const up=(best&&bd<M.spacing*0.45)?upOf(best):-0.3;
    if(best&&bd<M.spacing*0.45&&s.state==='대기'&&up<=0){const dcx=(best.box[0]+best.box[2])/2-s.cx;s.cx+=dcx*0.1;s.box=s.box.map((v,i)=>i%2===0?v+dcx*0.1:v)}
    if(up>0){if(s.holdStart==null)s.holdStart=t;s.lastPos=t;const dur=t-s.holdStart;s.holdDur=dur;s.state=dur>=HOLD_MIN?'만작':'준비'}
    else if(s.holdStart!=null&&t-s.lastPos>=tol){
      const dur=s.lastPos-s.holdStart;if(dur>=HOLD_MIN)motionRelease(s,s.lastPos,s.holdStart,dur);
      s.holdStart=null;s.lastPos=null;s.holdDur=0;if(s.state!=='발시')s.state='대기'}
  }
}
function motionRelease(s,end,hs,dur){
  const lo=Math.max(hs+1.0,end-WIN),hi=end+0.15,w=s.diffs.filter(d=>d[0]>=lo&&d[0]<=hi);
  let tM=end,method='자세';
  if(w.length>3){const mx=Math.max(...w.map(d=>d[1])),thr=Math.max(DIFF_MIN,FRAC*mx),f=w.find(d=>d[1]>thr);if(f){tM=f[0];method='동작'}}
  s.state='발시';setTimeout(()=>{if(s.state==='발시')s.state='대기'},1500);
  M.pend.push({idx:M.slots.indexOf(s),no:s.no,tM,dur,method});
  G().log('note',`${s.no}번 동작 발시 (만작 ${dur.toFixed(1)}초, ${method}) · 소리와 맞추는 중`,tM);
}
/* ---------- 소리 결합 ---------- */
M.onSoundRel=(t,info)=>{M.snd.push({t,info,cancel:false,used:false});while(M.snd.length&&t-M.snd[0].t>30)M.snd.shift()};
M.cancelSound=tTone=>{for(const e of M.snd)if(!e.used&&tTone-e.t>0&&tTone-e.t<=TONE_POST)e.cancel=true};
M.pcm=(t,pcm,nc)=>{   // keep ~4 s of mono sound (time of the last sample = t)
  const n=pcm.length/nc,m=new Float32Array(n);for(let i=0;i<n;i++){let a=0;for(let c=0;c<nc;c++)a+=pcm[i*nc+c];m[i]=a/nc}
  M.ringRate=G().S.pcmRate;M.ring.push({t,m});let tot=0;for(const r of M.ring)tot+=r.m.length;while(tot>M.ringRate*5&&M.ring.length>1){tot-=M.ring[0].m.length;M.ring.shift()}
};
M.onTone=t=>{if($('gmTone')&&$('gmTone').checked)M.tones.push({t,done:false})};
function segment(t0,t1){   // samples between t0..t1 (audio clock), or null
  const R=M.ringRate,out=[];let start=null;
  for(const r of M.ring){const rt1=r.t,rt0=r.t-r.m.length/R;if(rt1<t0||rt0>t1)continue;
    const a=Math.max(0,Math.round((t0-rt0)*R)),b=Math.min(r.m.length,Math.round((t1-rt0)*R));if(b>a){if(start==null)start=rt0+a/R;out.push(r.m.subarray(a,b))}}
  if(!out.length)return null;let n=0;for(const o of out)n+=o.length;const x=new Float32Array(n);let p=0;for(const o of out){x.set(o,p);p+=o.length}return x}
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]]}}
  for(let len=2;len<=n;len<<=1){const ang=-2*Math.PI/len,wr=Math.cos(ang),wi=Math.sin(ang);
    for(let i=0;i<n;i+=len){let cr=1,ci=0;for(let j=0;j<len/2;j++){const a=i+j,b=a+len/2,tr=re[b]*cr-im[b]*ci,ti=re[b]*ci+im[b]*cr;re[b]=re[a]-tr;im[b]=im[a]-ti;re[a]+=tr;im[a]+=ti;const nr=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nr}}}}
/* A형(우리 관): 1294→1535→679 Hz, B형(옆 관): ~800→673→1330→673 Hz. 1500~1600 Hz 가 두 창 이상이면 A */
M.classify=t=>{
  const x=segment(t-0.15,t+1.3);if(!x||x.length<M.ringRate*0.5)return{k:'?',why:'소리 부족'};
  const R=M.ringRate,N=1024,hop=Math.round(R*0.05),pk=[];let mx=0;
  for(let s=0;s+N<=x.length;s+=hop){const re=new Float64Array(N),im=new Float64Array(N);for(let i=0;i<N;i++)re[i]=x[s+i]*(0.5-0.5*Math.cos(2*Math.PI*i/N));fft(re,im);
    let bi=0,bm=0;for(let k=Math.floor(400*N/R);k<=Math.floor(3000*N/R);k++){const a=re[k]*re[k]+im[k]*im[k];if(a>bm){bm=a;bi=k}}
    pk.push([bi*R/N,bm]);if(bm>mx)mx=bm}
  let nA=0,nB=0;for(const[f,a]of pk){if(a<mx*0.05)continue;if(f>=1480&&f<=1600)nA++;else if(f>=1290&&f<=1350)nB++}
  return{k:nA>=2?'A':(nB>=2?'B':'?'),nA,nB};
};
/* ---------- 매 프레임 ---------- */
M.frame=t=>{
  const g=G();if(!g.S.running||!M.slots.length||!M.sess)return;
  if(!dc.width)setupDiff();diffFrame(t);
  if(!M.busy&&t-M.lastPoseT>=0.25){M.lastPoseT=t;inferAt(t)}
  if(M.inferMs.length){const m=M.inferMs.reduce((a,b)=>a+b)/M.inferMs.length;st(`자세 ${m.toFixed(0)}ms/회`+(m>250?' · 느림(동작 시각이 늦어질 수 있음)':''))}
};
M.tick=t=>{
  const g=G();
  // 동작 발시 ↔ 소리 발시 결합 (소리는 뻐꾸기로 취소될 수 있어 RESOLVE_LAG 뒤에 확정)
  for(let i=M.pend.length-1;i>=0;i--){const p=M.pend[i];if(t<p.tM+RESOLVE_LAG)continue;M.pend.splice(i,1);
    let best=null;for(const e of M.snd){if(e.used||e.cancel)continue;const d=Math.abs(e.t-p.tM);if(d<=SND_MATCH&&(!best||d<Math.abs(best.t-p.tM)))best=e}
    const S=g.S,sl=M.slots[p.idx];
    // 이미 발시한 선수는 다음 차례까지 잠금: 발시 뒤 손이 올라가도 새 발시로 세지 않는다
    if(sl&&sl.lastRelT!=null&&p.tM-sl.lastRelT<LOCK){g.log('note',`${p.no}번은 방금 발시해서 동작 무시 (${(p.tM-sl.lastRelT).toFixed(1)}초 전)`,p.tM);continue}
    // 소리 없는 동작이 차례가 아닌 선수에게서 나오면 잡음으로 본다
    if(!best&&p.idx!==S.ptr.p){g.log('note',`${p.no}번 동작은 차례(${S.ptr.p+1}번)도 아니고 발시음도 없어 제외`,p.tM);continue}
    if(best)best.used=true;if(p.idx!==S.ptr.p){g.log('note',`순서 다름: 차례 ${S.ptr.p+1}번, 동작은 ${p.no}번 · 동작 기준으로 맞춤`,p.tM);S.ptr.p=p.idx}
    if(sl)sl.lastRelT=best?best.t:p.tM;
    if(best)g.onRelease(best.t,'sound',{...best.info,motion:+(p.tM-best.t).toFixed(2)});
    else{g.onRelease(p.tM,'motion',{rr:0});const a=S.arrows[S.arrows.length-1];if(a)a.flags.push('소리 미확인')}
  }
  // 뻐꾸기 종류 판별
  for(const e of M.tones){if(e.done||t<e.t+TONE_WAIT)continue;e.done=true;const r=M.classify(e.t);
    if(r.k==='A'){g.log('hit','우리 관 뻐꾸기(A형)',e.t);g.onFlash(e.t,'tone',{})}
    else if(r.k==='B')g.log('note','옆 관 뻐꾸기(B형) · 제외',e.t);
    else g.log('note','뻐꾸기 종류 불명 · 제외 '+(r.why||`(A${r.nA}/B${r.nB})`),e.t)}
  while(M.tones.length&&M.tones[0].done&&t-M.tones[0].t>10)M.tones.shift();
};
M.resetState=()=>{for(const s of M.slots)s.lastRelT=null;M.snd=[];M.pend=[];M.tones=[];M.prev=null;for(const s of M.slots){s.diffs=[];s.holdStart=null;s.lastPos=null;s.state='대기';s.holdDur=0}};
M.draw=(ovx,W,H,view)=>{
  if(!M.slots.length)return;const v=V(),dpr=devicePixelRatio,sx=W/v.videoWidth*view.z,sy=H/v.videoHeight*view.z,ox=view.tx*dpr,oy=view.ty*dpr;
  ovx.lineWidth=2*dpr;ovx.font=`bold ${13*dpr}px sans-serif`;
  for(const s of M.slots){const c=s.state==='발시'?'#ff5a5a':s.state==='만작'?'#FFC53D':s.state==='준비'?'#ee4':'#8c8';ovx.strokeStyle=c;ovx.fillStyle=c;
    const b=s.box;ovx.strokeRect(ox+b[0]*sx,oy+b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);ovx.fillText(s.no+'번 '+s.state,ox+b[0]*sx+4,oy+b[1]*sy+15*dpr)}
};
/* ---------- UI ---------- */
window.addEventListener('DOMContentLoaded',()=>{
  const cb=$('gmOn');if(!cb)return;
  cb.addEventListener('change',()=>{M.on=cb.checked;$('gmBox').hidden=!M.on;if(M.on)M.init();else st('');G()&&G().updateGo()});
  $('gmCal').addEventListener('click',M.calibrate);
});
})();
