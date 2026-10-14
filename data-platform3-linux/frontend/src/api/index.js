import axios from 'axios'
import { ElMessage } from 'element-plus'

// 统一 API 实例：Cookie 携带凭证；统一错误提示；401 自动登出
const api = axios.create({ baseURL: '/api', timeout: 60000, withCredentials: true })

let onUnauthorized = null
export function setUnauthorizedHandler(fn) { onUnauthorized = fn }

api.interceptors.response.use(
  (res) => res.data,
  async (err) => {
    const res = err.response
    const msg = res?.data?.message || res?.data?.detail || err.message || '请求失败'
    if (res?.status === 401 && !err.config.url.includes('/auth/login')) {
      // 尝试刷新一次 Token
      if (!err.config._retried) {
        err.config._retried = true
        try {
          await axios.post('/api/auth/refresh', {}, { withCredentials: true })
          return api(err.config)
        } catch { /* 刷新失败 → 登出 */ }
      }
      if (onUnauthorized) onUnauthorized(msg)
    } else if (!err.config.silent) {
      ElMessage.error(msg)
    }
    return Promise.reject(Object.assign(err, { friendlyMessage: msg }))
  }
)

export default api
