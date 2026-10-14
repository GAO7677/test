import assert from 'node:assert/strict'
import { decodeLabel, encodeLabel, encodeVideoLabel } from '../frontend/src/annotationAdapter.js'
let n=0
function check(name,fn){fn();n++;console.log('PASS '+name)}
check('图像框与caption保留',()=>{const a=decodeLabel({caption:'plane',boxes:[{x:1,y:2,w:3,h:4,label:'plane',confidence:.8}]},'image');assert.deepEqual(a.items[0].xyxy,[1,2,4,6]);const b=encodeLabel(a,a.items);assert.equal(b.caption,'plane');assert.equal(b.boxes[0].confidence,.8)})
check('视频空帧、帧号和元数据保留',()=>{const a=decodeLabel({fps:30,truncated:true,items:[{frame:0,time:0,boxes:[]},{frame:60,time:2,boxes:[{x:1,y:2,w:3,h:4,label:'plane'}]}]},'video');assert.equal(a.items[0].frame,60);assert.equal(a.items[0].type,'video_box');a.items[0].label='airplane';const b=encodeLabel(a,a.items);assert.equal(b.items.length,2);assert.equal(b.items[0].boxes.length,0);assert.equal(b.items[1].boxes[0].label,'airplane');assert.equal(b.truncated,true)})
check('语音片段可编辑文本',()=>{const a=decodeLabel({segments:[{start:0,end:1,text:'旧',confidence:.9}],sample_rate:16000},'voice');assert.equal(a.items[0].type,'segment');a.items[0].text='新';const b=encodeLabel(a,a.items);assert.equal(b.segments[0].text,'新');assert.equal(b.sample_rate,16000)})
check('实体codepoint转UTF16',()=>{const a=decodeLabel({entities:[{start:3,end:5,type:'地名'}]},'text','😀\r\n北京');assert.equal(a.items[0].start,4);assert.equal(a.items[0].text,'北京');assert.equal(encodeLabel(a,a.items).entities[0].offset_unit,'utf16')})
check('旧实体无偏移保留全部匹配',()=>{const a=decodeLabel({entities:[{entity:'北京',type:'地名'}],caption:'original'},'text','北京和北京');assert.deepEqual(a.items.map(a=>a.start),[0,3]);assert.equal(encodeLabel(a,a.items).caption,'original')})
check('雷达点、中心、框均保留',()=>{const a=decodeLabel({detections:[[1,2,3],{center:[4,5,6],label:'簇'},{type:'bev_box',range:[1,2],azimuth:[3,4],label:'目标'}]},'radar');assert.equal(a.items.length,3);assert.deepEqual(encodeLabel(a,a.items).detections[0].center,[1,2,3])})
check('未知坐标和损坏结果阻止覆盖',()=>{assert.throws(()=>decodeLabel({boxes:[{bbox:[1,2,3,4]}]},'image'));assert.throws(()=>decodeLabel({error:'failed'},'image'));assert.throws(()=>decodeLabel({entities:[{entity:'不存在',type:'地名'}]},'text','北京'))})
check('空标签允许新增且无串改原对象',()=>{const raw={boxes:[{x:1,y:2,w:3,h:4}]};const a=decodeLabel(raw,'image');a.items[0].xyxy[0]=99;assert.equal(raw.boxes[0].x,1);assert.deepEqual(encodeLabel(decodeLabel([],'text'),[]),[])})
check('仅有caption或类别的素材仍能添加标注',()=>{const a=decodeLabel({caption:'训练描述',category:'airplane'},'image');a.items.push({type:'bbox',xyxy:[1,2,3,4],label:'airplane'});const b=encodeLabel(a,a.items);assert.equal(b.caption,'训练描述');assert.equal(b.category,'airplane');assert.equal(b.annotations.length,1);assert.equal(decodeLabel(['飞机'],'image').items[0].label,'飞机')})
check('视频编码保留其他帧轨迹及输入对象',()=>{
  const raw={caption:'keep',annotations:[],custom:{keep:true}},state=decodeLabel(raw,'video')
  const untouched={type:'video_point',track_id:1,point_id:1,frame:1,xy:[7,8],visible:false,source:'cotracker3',trajectory_run:'run-1'}
  const rows=[{type:'video_box',track_id:1,frame:0,time:0,label:'bird',xyxy:[0,0,10,10],points_dirty:true,seed_status:'ready',tracking_points:[{point_id:1,xy:[4,5],origin:'manual'}]},
    {type:'video_point',track_id:1,point_id:1,frame:0,xy:[1,2],source:'cotracker3',trajectory_run:'run-1'},untouched]
  const before=JSON.stringify(rows),out=encodeVideoLabel(state,rows,[rows[0]],{export_types:['trajectory'],fps:25})
  assert.equal(out.caption,'keep');assert.deepEqual(out.custom,{keep:true});assert.deepEqual(out.annotations.find(a=>a.frame===1),untouched)
  const point=out.annotations.find(a=>a.type==='video_point'&&a.frame===0);assert.deepEqual(point.xy,[4,5]);assert.equal(point.trajectory_run,'run-1');assert.equal(point.source,'manual')
  assert.equal(out.annotations[0].points_dirty,undefined);assert.equal(out.annotations[0].seed_status,undefined);assert.equal(JSON.stringify(rows),before)
  out.manual_annotations[0].label='changed';assert.equal(rows[0].label,'bird')
})
check('仅改mask不重写轨迹，清空点只清除对应帧',()=>{
  const box={type:'video_box',track_id:2,frame:0,xyxy:[0,0,9,9],points_dirty:true,tracking_points:[]}
  const point={type:'video_point',track_id:2,frame:0,point_id:1,xy:[1,2],source:'cotracker3'}
  const other={...point,track_id:3};const rows=[box,point,other]
  assert.deepEqual(encodeVideoLabel(null,rows,[],{export_types:['mask']}).annotations.slice(1),[point,other])
  assert.deepEqual(encodeVideoLabel(null,rows,[],{export_types:['trajectory']}).annotations.filter(a=>a.type==='video_point'),[other])
})
console.log(n+' adapter checks passed')
