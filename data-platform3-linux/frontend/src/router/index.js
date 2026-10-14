import { createRouter, createWebHashHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'

// hash 路由：打包后由 FastAPI StaticFiles 直接托管，无需服务端重写
const routes = [
  { path: '/login', name: 'login', component: () => import('../views/LoginView.vue'), meta: { public: true } },
  {
    path: '/',
    component: () => import('../views/MainLayout.vue'),
    redirect: '/voice',
    children: [
      // 数据获取层
      { path: 'voice', component: () => import('../views/DataTypeView.vue'), props: { dataType: 'voice' } },
      { path: 'radar', component: () => import('../views/DataTypeView.vue'), props: { dataType: 'radar' } },
      { path: 'image', component: () => import('../views/DataTypeView.vue'), props: { dataType: 'image' } },
      { path: 'video', component: () => import('../views/DataTypeView.vue'), props: { dataType: 'video' } },
      { path: 'text', component: () => import('../views/DataTypeView.vue'), props: { dataType: 'text' } },
      { path: 'writeback', component: () => import('../views/WritebackView.vue') },
      { path: 'graph-fetch', component: () => import('../views/GraphFetchView.vue') },
      { path: 'ontology-fetch', component: () => import('../views/OntologyFetchView.vue') },
      // 数据预处理层
      { path: 'datasets', component: () => import('../views/DatasetView.vue') },
      { path: 'workflow', component: () => import('../views/WorkflowView.vue') },
      { path: 'fewshot', component: () => import('../views/FewshotView.vue') },
      { path: 'auto-label', component: () => import('../views/AutoLabelView.vue') },
      { path: 'manual-label', component: () => import('../views/ManualLabelView.vue') },
      { path: 'clean', component: () => import('../views/CleanView.vue') },
      { path: 'integrate', component: () => import('../views/IntegrateView.vue') },
      { path: 'transform', component: () => import('../views/TransformView.vue') },
      { path: 'sample', component: () => import('../views/SampleView.vue') },
      { path: 'graph-extract', component: () => import('../views/GraphExtractView.vue') },
      { path: 'ontology-extract', component: () => import('../views/OntologyExtractView.vue') },
      { path: 'download', component: () => import('../views/DownloadView.vue') },
      // 数据可视化与标准层
      { path: 'viz-center', component: () => import('../views/VizCenterView.vue') },
      { path: 'viz/:type/:id', component: () => import('../views/VizDetailView.vue'), props: true },
      { path: 'quality', component: () => import('../views/QualityPanel.vue') },
      { path: 'standards', component: () => import('../views/StandardView.vue') },
      { path: 'semantic', component: () => import('../views/SemanticView.vue') },
      // 系统管理
      { path: 'users', component: () => import('../views/UserManageView.vue') },
      { path: 'logs', component: () => import('../views/LogView.vue') },
      { path: 'settings', component: () => import('../views/SettingsView.vue') },
    ],
  },
]

const router = createRouter({ history: createWebHashHistory(), routes })

router.beforeEach(async (to) => {
  const auth = useAuthStore()
  if (!auth.loaded) await auth.fetchMe()
  if (to.meta.public) {
    if (auth.isLogin) return '/'
    return true
  }
  if (!auth.isLogin) return '/login'
  return true
})

export default router
