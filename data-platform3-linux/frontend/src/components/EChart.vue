<template>
  <div ref="el" :style="{ width: '100%', height }" />
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  option: { type: Object, required: true },
  height: { type: String, default: '320px' },
})
const el = ref(null)
let chart = null

const base = {
  backgroundColor: 'transparent',
  textStyle: { color: '#94a3b8' },
}

function render() {
  if (!chart && el.value) chart = echarts.init(el.value, null, { renderer: 'canvas' })
  if (chart) chart.setOption({ ...base, ...props.option }, true)
}

onMounted(() => {
  render()
  window.addEventListener('resize', resize)
})
onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
})
watch(() => props.option, render, { deep: true })
function resize() { chart?.resize() }
</script>
