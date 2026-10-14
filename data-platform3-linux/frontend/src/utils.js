// 前端公共工具与常量

export function fmtSize(bytes) {
  if (!bytes) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1)
  return (bytes / Math.pow(k, i)).toFixed(i ? 1 : 0) + ' ' + sizes[i]
}

export function fmtDuration(sec) {
  if (sec == null) return '-'
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export const TYPE_META = {
  voice: {
    icon: '🎙️', title: '语音获取', short: '语音数据',
    desc: '航空语音数据采集与管理，支持 WAV/MP3/FLAC/M4A/OGG 格式上传、分片断点续传与标注',
    formats: ['WAV', 'MP3', 'FLAC', 'M4A', 'OGG'],
    acceptDesc: 'WAV / MP3 / FLAC / M4A / OGG',
    columns: [
      { prop: 'duration_seconds', label: '时长', width: 90, fmt: fmtDuration },
      { prop: 'sample_rate', label: '采样率', width: 90, fmt: (v) => (v ? v / 1000 + 'k' : '-') },
    ],
  },
  radar: {
    icon: '📡', title: '雷达获取', short: '雷达数据',
    desc: '雷达点云数据采集与管理，支持 CSV/PLY 格式上传与点云解析',
    formats: ['CSV', 'PLY'],
    acceptDesc: 'CSV / PLY 点云',
    columns: [
      { prop: 'point_count', label: '点数', width: 110, fmt: (v) => (v ?? '-').toLocaleString?.() ?? v },
      { prop: 'scan_mode', label: '扫描模式', width: 100 },
    ],
  },
  image: {
    icon: '🖼️', title: '图像获取', short: '图像数据',
    desc: '航空图像数据采集与管理，支持 JPG/PNG/BMP/TIFF 格式上传与缩略图预览',
    formats: ['JPG', 'PNG', 'BMP', 'TIFF'],
    acceptDesc: 'JPG / PNG / BMP / TIFF',
    columns: [
      { prop: 'width', label: '分辨率', width: 120, fmt: (v, row) => v ? `${v}x${row?.height ?? ''}` : '-' },
    ],
  },
  video: {
    icon: '🎬', title: '视频获取', short: '视频数据',
    desc: '航空视频数据采集与管理，支持 MP4/AVI/MOV/MKV 格式上传与首帧预览',
    formats: ['MP4', 'AVI', 'MOV', 'MKV'],
    acceptDesc: 'MP4 / AVI / MOV / MKV',
    columns: [
      { prop: 'duration_seconds', label: '时长', width: 90, fmt: fmtDuration },
      { prop: 'fps', label: '帧率', width: 70 },
    ],
  },
  text: {
    icon: '📄', title: '文本获取', short: '文本数据',
    desc: '航空文本数据采集与管理，支持 TXT/CSV/JSON/XML 格式上传与内容预览',
    formats: ['TXT', 'CSV', 'JSON', 'XML'],
    acceptDesc: 'TXT / CSV / JSON / XML',
    columns: [
      { prop: 'line_count', label: '行数', width: 100, fmt: (v) => v ?? '-' },
      { prop: 'encoding', label: '编码', width: 80 },
    ],
  },
}

export const TYPE_NAMES = { voice: '语音', radar: '雷达', image: '图像', video: '视频', text: '文本', mixed: '综合' }

// 密级六级（升序）：公开 < 受控 < 内部 < 秘密 < 机密 < 绝密
export const SECRET_LEVELS = ['公开', '受控', '内部', '秘密', '机密', '绝密']
export const SECRET_TAGS = { 公开: 'success', 受控: 'info', 内部: 'primary', 秘密: 'warning', 机密: 'warning', 绝密: 'danger' }

// 自定义评估标准可用指标
export const QUALITY_METRICS = [
  { key: 'file_count', name: '文件总数' },
  { key: 'nonempty_rate', name: '非空文件率(%)' },
  { key: 'parseable_rate', name: '元数据可解析率(%)' },
  { key: 'content_pass_rate', name: '内容校验通过率(%)' },
  { key: 'format_ext_count', name: '扩展名种类数' },
  { key: 'named_rate', name: '命名规范率(%)' },
  { key: 'avg_age_days', name: '平均数据龄期(天)' },
  { key: 'label_coverage', name: '标注覆盖率(%)' },
  { key: 'readable_rate', name: '可读取率(%)' },
]

export const STAGES = ['原始', '已标注', '清洗后', '转换后', '集成后', '采样后']

export const TASK_TYPE_NAMES = {
  init: '初始版本', copy: '复制版本', auto_label: '自动标注', manual_label: '手动标注',
  hybrid_label: '模板标注', clean: '数据清洗', integrate: '数据集成',
  transform: '数据转换', sample: '数据采样',
}
