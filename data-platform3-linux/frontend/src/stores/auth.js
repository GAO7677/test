import { defineStore } from 'pinia'
import api from '../api'

// 角色权限矩阵（9.1）：visitor 仅查看；admin 数据/预处理 CRUD；system_admin 全部
export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    loaded: false,
  }),
  getters: {
    isLogin: (s) => !!s.user,
    role: (s) => s.user?.role || '',
    canWrite: (s) => ['system_admin', 'admin'].includes(s.user?.role),
    // 任务操作权限：管理员不受限；访客需 operate_level >= 0（后端同步强制校验）
    canOperate: (s) => ['system_admin', 'admin'].includes(s.user?.role)
      || (s.user?.role === 'visitor' && (s.user?.operate_level ?? -1) >= 0),
    isSysAdmin: (s) => s.user?.role === 'system_admin',
    canViewUsers: (s) => ['system_admin', 'admin'].includes(s.user?.role),
    canViewLogs: (s) => ['system_admin', 'admin'].includes(s.user?.role),
    mustChangePassword: (s) => !!s.user?.must_change_password,
  },
  actions: {
    async fetchMe() {
      try {
        this.user = await api.get('/auth/me')
      } catch {
        this.user = null
      } finally {
        this.loaded = true
      }
      return this.user
    },
    async login(username, password) {
      const data = await api.post('/auth/login', { username, password })
      this.user = data
      this.loaded = true
      return data
    },
    async logout() {
      try { await api.post('/auth/logout') } catch { /* ignore */ }
      this.user = null
    },
  },
})
