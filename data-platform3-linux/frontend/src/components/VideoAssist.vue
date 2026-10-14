<template>
  <div v-if="job" class="video-job">
    <span>{{ names[job.status] || job.status }}</span>
    <el-progress style="width:180px" :percentage="job.progress || 0" />
    <el-button v-if="busy && !sending" size="small" @click="cancel">取消任务</el-button>
    <span v-if="job.error_msg">{{ job.error_msg }}</span>
    <el-button v-if="pollError" size="small" @click="poll">重试获取状态</el-button>
  </div>
</template>
<script setup>
// One task lifecycle: submit, poll and cancel. The workbench owns result loading.
import {computed,onBeforeUnmount,ref,watch} from 'vue'
import api from '../api'
import {clone} from '../annotationAdapter'
const props=defineProps({disabled:Boolean})
const emit=defineEmits(['busy','submitted','status','completed'])
const job=ref(null),sending=ref(false),pollError=ref(false)
const busy=computed(()=>sending.value || ['pending','running'].includes(job.value?.status))
const names={pending:'排队中',running:'自动标注中',completed:'已完成',failed:'执行失败',cancelled:'已取消'}
watch(busy,value=>emit('busy',value),{immediate:true})
watch(job,value=>{if(value)emit('status',value)},{flush:'sync'})
let timer=null,disposed=false,opened=false,requestVersion=0
const current=version=>!disposed && version===requestVersion
async function poll(){
  clearTimeout(timer);pollError.value=false
  if(!job.value)return
  const version=requestVersion,id=job.value.id
  try{
    const result=await api.get(`/video-annotation/${id}`)
    if(!current(version) || job.value?.id!==id)return
    job.value=result
    if(result.status==='completed'&&!opened){opened=true;emit('completed',result)}
    else if(busy.value)timer=setTimeout(poll,1000)
  }catch{if(current(version))pollError.value=true}
}
async function run(payload){
  if(busy.value || props.disabled || disposed)return null
  const version=++requestVersion
  clearTimeout(timer);sending.value=true;opened=false;pollError.value=false
  try{
    const result=await api.post('/video-annotation',clone(payload))
    if(!current(version))return null
    job.value=result;emit('submitted',result);poll();return result
  }catch{return null}finally{if(current(version))sending.value=false}
}
async function cancel(){
  if(!job.value || sending.value)return
  const version=++requestVersion,id=job.value.id
  clearTimeout(timer);sending.value=true
  try{
    const result=await api.post(`/video-annotation/${id}/cancel`)
    if(current(version))job.value=result
  }catch{if(current(version))pollError.value=true}
  finally{if(current(version))sending.value=false}
}
defineExpose({run})
onBeforeUnmount(()=>{disposed=true;requestVersion++;clearTimeout(timer)})
</script>
<style scoped>.video-job{display:flex;align-items:center;gap:12px;margin:10px 0}</style>
