const { app } = window.comfyAPI.app
const { api } = window.comfyAPI.api

api.addEventListener('strict_h3_preview', ({ detail }) => {
	const ids = String(detail.node).split(':')
	let graph = app.graph
	for (const id of ids.slice(0, -1))
		graph = graph?.getNodeById(Number(id))?.subgraph
	const node = graph?.getNodeById(Number(ids.at(-1)))
	if (!node)
		return
	if (!node.strictPreview) {
		node.strictPreview = document.createElement('img')
		node.strictPreview.style.cssText = 'width: 100%; height: 100%; object-fit: contain'
		node.addDOMWidget('preview', 'strict_h3_preview', node.strictPreview, { serialize: false })
		node.setSize([Math.max(node.size[0], 320), Math.max(node.size[1], 520)])
	}
	node.strictPreview.src = 'data:image/webp;base64,' + detail.image
})
