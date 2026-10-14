import { ref } from 'vue'
import api from './api'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'

// 预处理页共享：数据集选择 + 版本映射
export function useDatasets() {
  const route = useRoute()
  const datasets = ref([])
  const versionMap = ref({})       // version_id -> V{n}
  const datasetVersions = ref({})  // dataset_id -> versions[]

  async function loadDatasets() {
    const items = []
    for (let page = 1; ; page++) {
      const data = await api.get('/datasets', { params: { page, page_size: 100 } })
      items.push(...data.items)
      if (!data.items.length || items.length >= data.total) break
    }
    datasets.value = items
    const requested=Number(route.query.dataset_id || route.query.dataset)
    if(requested&&!datasets.value.some(d=>d.id===requested))datasets.value.push(await api.get(`/datasets/${requested}`))
    // Bound concurrent requests; do not serialize one round trip per dataset.
    for (let i = 0; i < datasets.value.length; i += 8) {
      await Promise.all(datasets.value.slice(i, i + 8).map(async (ds) => {
        const vs = await api.get(`/datasets/${ds.id}/versions`)
        datasetVersions.value[ds.id] = vs.items
        for (const v of vs.items) versionMap.value[v.id] = v.version
      }))
    }
  }

  function currentVersionId(datasetId) {
    const ds = datasets.value.find((d) => d.id === datasetId)
    return ds?.current_version?.id
  }

  function routeInput() {
    const dataset_id=Number(route.query.dataset_id || route.query.dataset)
    if(!dataset_id || !datasets.value.some(d=>d.id===dataset_id))return null
    const requested=Number(route.query.version_id)
    if(requested&&!datasetVersions.value[dataset_id]?.some(v=>v.id===requested)){
      ElMessage.error('指定版本不属于此数据集，请重新选择');return null
    }
    return {dataset_id,version_id:requested||currentVersionId(dataset_id)}
  }

  return { datasets, versionMap, datasetVersions, loadDatasets, currentVersionId, routeInput }
}
