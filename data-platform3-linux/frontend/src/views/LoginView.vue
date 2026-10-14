<template>
  <div class="login-page">
    <div class="login-card">
      <div class="brand">
        <div class="brand-icon">✈</div>
        <h1>数据中台</h1>
        <p>语音 · 雷达 · 图像 · 视频 · 文本 全生命周期管理</p>
      </div>
      <el-form @submit.prevent="onLogin">
        <el-form-item>
          <el-input v-model="username" size="large" placeholder="用户名" :prefix-icon="User" autocomplete="username" />
        </el-form-item>
        <el-form-item>
          <el-input v-model="password" size="large" type="password" placeholder="密码"
                    :prefix-icon="Lock" show-password autocomplete="current-password"
                    @keyup.enter="onLogin" />
        </el-form-item>
        <el-alert v-if="errorMsg" :title="errorMsg" type="error" :closable="false" class="login-alert" />
        <el-button type="primary" size="large" class="login-btn" :loading="loading" @click="onLogin">
          登 录
        </el-button>
      </el-form>
    </div>

    <!-- 首次登录强制修改密码：不可通过 Esc / 点击遮罩 / 关闭按钮跳过 -->
    <el-dialog v-model="showForceChange" title="首次登录须修改密码" width="420px"
               :close-on-click-modal="false" :close-on-press-escape="false" :show-close="false">
      <el-alert type="warning" :closable="false" title="为保障账号安全，首次登录必须修改初始密码后方可进入系统"
                style="margin-bottom: 14px" />
      <el-form label-width="90px">
        <el-form-item label="当前密码">
          <el-input v-model="pw.old" type="password" show-password />
        </el-form-item>
        <el-form-item label="新密码">
          <el-input v-model="pw.next" type="password" show-password placeholder="至少 6 位" />
        </el-form-item>
        <el-form-item label="确认新密码">
          <el-input v-model="pw.confirm" type="password" show-password @keyup.enter="onChangePassword" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button type="primary" :loading="changing" @click="onChangePassword" style="width: 100%">
          确认修改并进入系统
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { useAuthStore } from '../stores/auth'
import api from '../api'

const router = useRouter()
const auth = useAuthStore()
const username = ref('')
const password = ref('')
const errorMsg = ref('')
const loading = ref(false)
const showForceChange = ref(false)
const changing = ref(false)
const pw = reactive({ old: '', next: '', confirm: '' })

async function onLogin() {
  if (!username.value || !password.value) {
    errorMsg.value = '请输入用户名和密码'
    return
  }
  loading.value = true
  errorMsg.value = ''
  try {
    const data = await auth.login(username.value, password.value)
    if (data.must_change_password) {
      pw.old = password.value
      showForceChange.value = true   // 强制改密，改密成功方可进入系统
    } else {
      router.push('/')
    }
  } catch (e) {
    errorMsg.value = e.friendlyMessage || '登录失败'
  } finally {
    loading.value = false
  }
}

async function onChangePassword() {
  if (pw.next.length < 6) return ElMessage.error('新密码长度至少 6 位')
  if (pw.next !== pw.confirm) return ElMessage.error('两次输入的新密码不一致')
  if (pw.next === pw.old) return ElMessage.error('新密码不得与当前密码相同')
  changing.value = true
  try {
    await api.post('/auth/change-password', { old_password: pw.old, new_password: pw.next })
    ElMessage.success('密码修改成功')
    auth.user.must_change_password = false
    showForceChange.value = false
    router.push('/')
  } catch { /* 拦截器已提示 */ } finally {
    changing.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background:
    radial-gradient(ellipse at 20% 20%, rgba(30, 58, 138, 0.5), transparent 55%),
    radial-gradient(ellipse at 80% 80%, rgba(96, 165, 250, 0.15), transparent 50%),
    #0c1220;
}
.login-card {
  width: 400px;
  background: rgba(19, 28, 49, 0.9);
  border: 1px solid #24304d;
  border-radius: 14px;
  padding: 36px 34px 30px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}
.brand { text-align: center; margin-bottom: 26px; }
.brand-icon {
  width: 56px; height: 56px; margin: 0 auto 12px;
  background: linear-gradient(135deg, #1e3a8a, #60a5fa);
  border-radius: 14px; font-size: 1.7rem;
  display: flex; align-items: center; justify-content: center;
}
.brand h1 { font-size: 1.25rem; margin: 0 0 6px; }
.brand p { color: #94a3b8; font-size: 0.78rem; margin: 0; }
.login-btn { width: 100%; margin-top: 6px; }
.login-alert { margin-bottom: 14px; }
</style>
