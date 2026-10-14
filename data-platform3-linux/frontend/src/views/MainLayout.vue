<template>
  <div class="layout" :class="{ 'fewshot-layout': route.path === '/fewshot' }">
    <!-- 顶部导航 -->
    <header class="top-header">
      <div class="logo">
        <span class="logo-icon">✈</span>
        <span class="logo-text">数据中台</span>
      </div>
      <div class="header-right">
        <el-button v-if="route.path === '/fewshot'" class="mobile-nav-toggle" text aria-label="展开导航" @click="mobileNav = !mobileNav">菜单</el-button>
        <span class="env-tag">{{ route.path === '/fewshot' ? '本地训练' : '离线环境' }}</span>
        <el-dropdown trigger="click" @command="onUserCommand">
          <span class="user-chip">
            <el-icon><User /></el-icon>
            {{ auth.user?.display_name || auth.user?.username }}
            <el-tag size="small" effect="dark" :type="roleTagType">{{ auth.user?.role_name }}</el-tag>
            <el-icon><ArrowDown /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="password">
                <el-icon><Key /></el-icon>修改密码
              </el-dropdown-item>
              <el-dropdown-item command="logout" divided>
                <el-icon><SwitchButton /></el-icon>退出登录
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </header>

    <div class="body">
      <!-- 左侧菜单（按角色动态渲染） -->
      <aside class="sidebar" :class="{ 'mobile-open': mobileNav }">
        <div class="sidebar-title">架构目录</div>
        <div v-for="group in menus" :key="group.id" class="nav-group">
          <div class="nav-parent" :class="{ active: isGroupActive(group) }" @click="toggle(group.id)">
            <span class="nav-icon">{{ group.icon }}</span>
            <span class="nav-title">{{ group.title }}</span>
            <span class="nav-arrow" :class="{ open: openGroups.includes(group.id) }">▶</span>
          </div>
          <div v-show="openGroups.includes(group.id)" class="nav-children">
            <router-link v-for="child in group.children" :key="child.path" :to="child.path"
                         class="nav-child" @click="mobileNav = false" :class="{ active: route.path === child.path }">
              <span class="dot"></span>{{ child.title }}
            </router-link>
          </div>
        </div>
      </aside>

      <!-- 内容区 -->
      <main class="content">
        <div v-if="route.query.download_request_id && !['/fewshot','/download'].includes(route.path)" class="download-return">
          <el-button text @click="router.push({path:'/download',query:{download_request_id:route.query.download_request_id}})">← 返回对应的数据下载</el-button>
        </div>
        <router-view v-slot="{ Component }">
          <component :is="Component" :key="route.fullPath" />
        </router-view>
      </main>
    </div>

    <!-- 自助修改密码 -->
    <el-dialog v-model="showPw" title="修改密码" width="420px">
      <el-form label-width="90px">
        <el-form-item label="当前密码"><el-input v-model="pw.old" type="password" show-password /></el-form-item>
        <el-form-item label="新密码"><el-input v-model="pw.next" type="password" show-password placeholder="至少 6 位" /></el-form-item>
        <el-form-item label="确认新密码"><el-input v-model="pw.confirm" type="password" show-password /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showPw = false">取消</el-button>
        <el-button type="primary" :loading="pwLoading" @click="onChangePw">确认修改</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '../stores/auth'
import api, { setUnauthorizedHandler } from '../api'

const route = useRoute()
const router = useRouter()
const mobileNav = ref(false)
const auth = useAuthStore()

const openGroups = ref(['collection', 'processing', 'visualization', 'system'])
function toggle(id) {
  const i = openGroups.value.indexOf(id)
  if (i >= 0) openGroups.value.splice(i, 1)
  else openGroups.value.push(id)
}

const menus = computed(() => {
  const base = [
    { id: 'collection', icon: '📥', title: '数据获取层', children: [
      { path: '/voice', title: '语音获取' }, { path: '/radar', title: '雷达获取' },
      { path: '/image', title: '图像获取' }, { path: '/video', title: '视频获取' },
      { path: '/text', title: '文本获取' }, { path: '/writeback', title: '生成结果回写' },
      { path: '/graph-fetch', title: '知识图谱获取' }, { path: '/ontology-fetch', title: '本体获取' },
    ]},
    { id: 'processing', icon: '⚙️', title: '数据预处理层', children: [
      { path: '/datasets', title: '数据集管理' }, { path: '/workflow', title: '工作流' },
      { path: '/fewshot', title: '少样本数据生成' },
      { path: '/auto-label', title: '规则自动标注' }, { path: '/manual-label', title: '手动标注工具' },
      { path: '/clean', title: '数据清洗' },
      { path: '/integrate', title: '数据集成' }, { path: '/transform', title: '数据转换' },
      { path: '/sample', title: '数据采样' },
      { path: '/graph-extract', title: '知识图谱抽取' }, { path: '/ontology-extract', title: '本体抽取' },
      { path: '/download', title: '数据下载' },
    ]},
    { id: 'visualization', icon: '📊', title: '数据可视化与标准层', children: [
      { path: '/viz-center', title: '数据可视化中心' },
      { path: '/quality', title: '数据质量评估' },
      { path: '/standards', title: '参考标准管理' },
      { path: '/semantic', title: '统一语义管理' },
    ]},
  ]
  if (auth.canViewUsers || auth.canViewLogs) {
    const children = []
    if (auth.canViewUsers) children.push({ path: '/users', title: '用户管理' })
    if (auth.canViewLogs) children.push({ path: '/logs', title: '操作日志' })
    if (auth.isSysAdmin) children.push({ path: '/settings', title: '系统配置' })
    base.push({ id: 'system', icon: '🛡️', title: '系统管理', children })
  }
  return base
})

function isGroupActive(group) {
  return group.children.some((c) => route.path.startsWith(c.path))
}

const roleTagType = computed(() =>
  ({ system_admin: 'danger', admin: 'warning', visitor: 'info' }[auth.role] || 'info'))

// ---- 用户菜单 ----
const showPw = ref(false)
const pwLoading = ref(false)
const pw = reactive({ old: '', next: '', confirm: '' })

function onUserCommand(cmd) {
  if (cmd === 'logout') {
    ElMessageBox.confirm('确认退出登录？', '退出登录', { type: 'warning' }).then(async () => {
      await auth.logout()
      router.push('/login')
    }).catch(() => {})
  } else if (cmd === 'password') {
    pw.old = pw.next = pw.confirm = ''
    showPw.value = true
  }
}

async function onChangePw() {
  if (pw.next.length < 6) return ElMessage.error('新密码长度至少 6 位')
  if (pw.next !== pw.confirm) return ElMessage.error('两次输入的新密码不一致')
  pwLoading.value = true
  try {
    await api.post('/auth/change-password', { old_password: pw.old, new_password: pw.next })
    ElMessage.success('密码修改成功')
    showPw.value = false
  } catch { /* 已提示 */ } finally {
    pwLoading.value = false
  }
}

// 401（会话超时/被挤下线）→ 自动登出回登录页
setUnauthorizedHandler((msg) => {
  auth.user = null
  ElMessage.warning(msg || '会话已失效，请重新登录')
  router.push('/login')
})
</script>

<style scoped>
.layout { height: 100vh; display: flex; flex-direction: column; }
.top-header {
  height: 56px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px;
  background: linear-gradient(90deg, #101a30, #16223c);
  border-bottom: 1px solid #24304d;
}
.logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; }
.logo-icon {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, #1e3a8a, #60a5fa);
  display: flex; align-items: center; justify-content: center; font-size: 1rem;
}
.header-right { display: flex; align-items: center; gap: 14px; }
.env-tag {
  font-size: 0.72rem; color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4);
  padding: 2px 8px; border-radius: 10px;
}
.user-chip {
  display: flex; align-items: center; gap: 7px; cursor: pointer;
  color: #e2e8f0; font-size: 0.88rem;
}
.body { flex: 1; display: flex; overflow: hidden; }
.sidebar {
  width: 220px; flex-shrink: 0; overflow-y: auto;
  background: #0f1729; border-right: 1px solid #1e2a45; padding: 10px 8px;
}
.sidebar-title {
  color: #64748b; font-size: 0.72rem; padding: 6px 12px 10px;
  letter-spacing: 1px;
}
.nav-parent {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 12px; border-radius: 8px; cursor: pointer;
  color: #cbd5e1; font-size: 0.88rem; user-select: none;
}
.nav-parent:hover { background: #1a2540; }
.nav-parent.active { color: #60a5fa; }
.nav-arrow { margin-left: auto; font-size: 0.6rem; transition: transform 0.2s; color: #64748b; }
.nav-arrow.open { transform: rotate(90deg); }
.nav-children { padding: 2px 0 6px; }
.nav-child {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px 8px 34px; border-radius: 8px;
  color: #94a3b8; font-size: 0.83rem; text-decoration: none;
}
.nav-child:hover { background: #1a2540; color: #cbd5e1; }
.nav-child.active { background: rgba(96, 165, 250, 0.12); color: #60a5fa; }
.dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; opacity: 0.6; }
.content { flex: 1; overflow-y: auto; }
.mobile-nav-toggle { display: none; }
@media (max-width: 760px) {
  .fewshot-layout .mobile-nav-toggle { display: inline-flex; }
  .fewshot-layout .sidebar { display: none; }
  .fewshot-layout .sidebar.mobile-open { display: block; position: fixed; top: 56px; bottom: 0; left: 0; z-index: 100; }
  .fewshot-layout .content { min-width: 0; }
  .fewshot-layout .env-tag, .fewshot-layout .user-chip .el-tag { display: none; }
  .fewshot-layout .top-header { padding: 0 12px; }
  .fewshot-layout .header-right { gap: 8px; }
}
</style>
