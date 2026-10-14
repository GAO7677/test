// Small adapters preserve the original container and metadata on save.
export const clone = (value) => JSON.parse(JSON.stringify(value))
export function normalizeAnn(value, modality, text = '', fps = 25) {
  if (typeof value === 'string') return { type: 'classification', label: value }
  if (Array.isArray(value)) return { type: 'point', center: value, label: '检测点' }
  if (!value || typeof value !== 'object') throw new Error('不支持的标注条目，请先检查导入格式')
  const a = clone(value)
  if (a.bbox && !a.xyxy) throw new Error('bbox 未声明坐标格式，请转换为 xyxy 后再编辑')
  if ([a.x, a.y, a.w, a.h].every(Number.isFinite)) {
    a.xyxy = [a.x, a.y, a.x + a.w, a.y + a.h]
    delete a.x; delete a.y; delete a.w; delete a.h
  }
  if (a.xyxy) a.type = modality === 'video' ? 'video_box' : 'bbox'
  if (modality === 'voice' && a.start != null && a.end != null) a.type = 'segment'
  if (modality === 'text' && Number.isInteger(a.start) && Number.isInteger(a.end)) {
    a.label ||= a.type === 'span' ? '实体' : a.type
    a.type = 'span'
    if (a.offset_unit !== 'utf16') {
      a.start = [...text].slice(0, a.start).join('').length
      a.end = [...text].slice(0, a.end).join('').length
    }
    a.offset_unit = 'utf16'
    a.text = text.slice(a.start, a.end)
  }
  if (modality === 'video' && a.xyxy) {
    a.frame = a.frame ?? Math.round((a.time || 0) * fps)
    a.time = a.time ?? a.frame / fps
  }
  return a
}
export function decodeLabel(raw, modality, text = '') {
  if (raw?.error || raw?.requires_relabel) throw new Error('此文件的自动标注失败，请重新标注后再编辑')
  const keys = ['annotations', 'items', 'boxes', 'segments', 'entities', 'detections']
  let key = Array.isArray(raw) ? null : keys.find((k) => Array.isArray(raw?.[k]))
  if (!key && raw && ['caption','category','classification','sentiment'].some((k) => k in raw)) key = 'annotations'
  if (!Array.isArray(raw) && !key) throw new Error('当前标签不是可编辑的框、片段或实体格式，已保留原文件')
  const source = key ? (raw[key] || []) : raw
  const fps = raw?.fps || 25
  const nestedVideo = modality === 'video' && source.some((a) => Array.isArray(a?.boxes))
  let rows = nestedVideo ? source.flatMap((frame) => Array.isArray(frame.boxes)
    ? frame.boxes.map((box) => ({ ...box, frame: frame.frame, time: frame.time })) : [frame]) : source
  // Older rule entities have no offsets. Materialize each exact occurrence.
  if (modality === 'text') rows = rows.flatMap((a) => {
    if (Number.isInteger(a.start) || !(a.entity || a.text)) return [a]
    const word = a.entity || a.text, matches = []
    for (let at = text.indexOf(word); at >= 0; at = text.indexOf(word, at + word.length))
      matches.push({ ...a, start: at, end: at + word.length, offset_unit: 'utf16', label: a.label || a.type, type: 'span' })
    if (!matches.length) throw new Error('实体在当前文本中不存在，已阻止覆盖原标注')
    return matches
  })
  return { items: rows.map((a) => normalizeAnn(a, modality, text, fps)), original: clone(raw), key, nestedVideo, fps }
}
export function encodeLabel(state, rows) {
  const items = clone(rows)
  if (!state || Array.isArray(state.original)) return items
  const out = clone(state.original)
  if (state.nestedVideo) {
    const frames = out[state.key].filter((a) => Array.isArray(a.boxes)).map((a) => ({ ...a, boxes: [] }))
    const other = []
    for (const a of items) {
      if (!a.xyxy) { other.push(a); continue }
      let frame = frames.find((f) => f.frame === a.frame)
      if (!frame) { frame = { frame: a.frame, time: a.time, boxes: [] }; frames.push(frame) }
      frame.boxes.push(a)
    }
    for (const f of frames) { f.label = `${f.boxes.length} 个目标`; f.status = f.boxes.length ? 'detected' : 'empty' }
    out[state.key] = [...frames.sort((a,b) => a.frame-b.frame), ...other]
    out.item_count = out[state.key].length
  } else {
    out[state.key] = items
    for (const k of ['box_count','segment_count','entity_count','item_count','target_count'])
      if (k in out) out[k] = items.length
  }
  out.review_status = 'human_edited'
  return out
}


export const videoAnnotationKey = a => `${a.track_id}:${a.frame}`

// Encode edited video seeds without rewriting points on untouched objects/frames.
export function encodeVideoLabel(state, annotations, manualAnnotations, metadata) {
  const outputs=metadata.export_types || []
  const edited=new Set(annotations.filter(a=>a.xyxy && a.points_dirty && outputs.includes('trajectory')).map(videoAnnotationKey))
  const rows=clone(annotations.filter(a=>a.type!=='video_point' || !edited.has(videoAnnotationKey(a))))
  for(const a of rows){delete a.seed_status;delete a.seed_error;delete a.points_dirty}
  const pointKey=a=>`${videoAnnotationKey(a)}:${a.point_id}`
  const previousPoints=new Map(annotations.filter(a=>a.type==='video_point').map(a=>[pointKey(a),a]))
  for(const a of rows.filter(a=>a.xyxy && edited.has(videoAnnotationKey(a)))){
    for(const p of a.tracking_points || []){
      const previous=previousPoints.get(pointKey({...a,point_id:p.point_id}))
      rows.push({...previous,trajectory_run:previous?.trajectory_run || `manual:${a.track_id}`,type:'video_point',track_id:a.track_id,point_id:p.point_id,xy:p.xy,label:a.label,frame:a.frame,time:a.time,visible:true,outside:false,source:'manual',seed_origin:p.origin,review_status:'human_edited'})
    }
  }
  const raw=encodeLabel(state,rows)
  return {...(Array.isArray(raw)?{}:raw),...clone(metadata),annotations:rows,manual_annotations:clone(manualAnnotations),review_status:'pending'}
}
