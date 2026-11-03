// Visit and image counters for MossWeb.html
const crypto = require("crypto"), fs = require("fs"), http = require("http"), path = require("path");
const folder = process.env.STATS_FOLDER || path.join(__dirname, "MossWebStats");
const port = Number(process.env.PORT || 8790);
fs.mkdirSync(folder, { recursive: true });
const file = name => path.join(folder, name);
const saltFile = file("salt.txt");
if (!fs.existsSync(saltFile)) fs.writeFileSync(saltFile, crypto.randomBytes(32).toString("hex"));
const salt = fs.readFileSync(saltFile, "utf8");
const lines = name => fs.existsSync(file(name)) ? fs.readFileSync(file(name), "utf8").split("\n").filter(line => line) : [];
const stats = { visits: lines("visits.txt").length, generated: 0, seconds: 0 };
for (const line of lines("generated.txt")) {
	stats.generated++;
	stats.seconds += Number(JSON.parse(line).seconds) || 0;
}
const summary = () => "Visits: " + stats.visits + ", Images Generated: " + stats.generated + ", Average Time: " +
	(stats.generated ? stats.seconds / stats.generated : 0).toFixed(1) + "s";
const visitor = request => {
	const forwarded = (request.headers["x-forwarded-for"] || "").split(",")[0].trim();
	const address = (forwarded || request.socket.remoteAddress || "").replace(/:\d+$/, "");
	return crypto.createHash("sha256").update(salt + address).digest("hex").slice(0, 16);
};
const text = (value, limit) => String(value ?? "").replace(/\s+/g, " ").trim().slice(0, limit);
const body = request => new Promise((resolve, reject) => {
	let received = "";
	request.on("data", part => {
		received += part;
		if (received.length > 8192) reject(new Error("too large"));
	});
	request.on("end", () => resolve(received));
});
const answer = (response, status, content) => {
	response.writeHead(status, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" });
	response.end(content);
};
http.createServer(async (request, response) => {
	try {
		const action = request.url.split("?")[0].split("/").pop();
		if (action === "visit" && request.method === "POST") {
			fs.appendFileSync(file("visits.txt"), new Date().toISOString() + " " + visitor(request) + "\n");
			stats.visits++;
		} else if (action === "generated" && request.method === "POST") {
			const sent = JSON.parse(await body(request));
			const seconds = Number(sent.seconds);
			if (!(seconds > 0 && seconds < 3600))
				return answer(response, 400, "seconds must be between 0 and 3600");
			const entry = { time: new Date().toISOString(), visitor: visitor(request), seconds: Math.round(seconds * 10) / 10,
				model: text(sent.model, 200), vae: text(sent.vae, 200), loras: text(sent.loras, 1000), prompt: text(sent.prompt, 20) };
			fs.appendFileSync(file("generated.txt"), JSON.stringify(entry) + "\n");
			stats.generated++;
			stats.seconds += entry.seconds;
		} else if (action !== "get_visits" || request.method !== "GET")
			return answer(response, 404, "unknown");
		answer(response, 200, summary());
	} catch (error) {
		answer(response, 400, "bad request");
	}
}).listen(port, "127.0.0.1", () => console.log("MossWebStats on port " + port + ", " + summary()));
