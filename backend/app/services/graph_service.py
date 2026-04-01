from collections import defaultdict, deque


class GraphService:
    def build_graph(self, transactions: list[dict]) -> tuple[list[dict], list[dict]]:
        user_nodes = {}
        device_nodes = {}
        ip_nodes = {}
        edges = set()

        for tx in transactions:
            u = f"user:{tx['user_id']}"
            d = f"device:{tx['device_id']}"
            ip = f"ip:{tx['ip_address']}"

            user_nodes[u] = {"id": u, "label": tx["user_id"], "type": "user"}
            device_nodes[d] = {"id": d, "label": tx["device_id"], "type": "device"}
            ip_nodes[ip] = {"id": ip, "label": tx["ip_address"], "type": "ip"}

            edges.add((u, d, "uses_device"))
            edges.add((u, ip, "uses_ip"))

        nodes = list(user_nodes.values()) + list(device_nodes.values()) + list(ip_nodes.values())
        edge_objs = [{"source": s, "target": t, "relation": r} for (s, t, r) in edges]
        return nodes, edge_objs

    def suspicious_component_count(self, nodes: list[dict], edges: list[dict]) -> int:
        adjacency = defaultdict(set)
        for e in edges:
            adjacency[e["source"]].add(e["target"])
            adjacency[e["target"]].add(e["source"])

        node_ids = {n["id"] for n in nodes}
        visited = set()
        suspicious = 0

        for node in node_ids:
            if node in visited:
                continue
            queue = deque([node])
            visited.add(node)
            component = set([node])

            while queue:
                cur = queue.popleft()
                for nxt in adjacency[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        component.add(nxt)
                        queue.append(nxt)

            users = [x for x in component if x.startswith("user:")]
            devices = [x for x in component if x.startswith("device:")]
            ips = [x for x in component if x.startswith("ip:")]
            if len(users) >= 3 and (len(devices) >= 1 or len(ips) >= 1):
                suspicious += 1

        return suspicious

    def shared_resource_pressure(self, transactions: list[dict], tx: dict) -> int:
        users_by_device = defaultdict(set)
        users_by_ip = defaultdict(set)

        for t in transactions:
            users_by_device[t["device_id"]].add(t["user_id"])
            users_by_ip[t["ip_address"]].add(t["user_id"])

        device_users = len(users_by_device[tx["device_id"]])
        ip_users = len(users_by_ip[tx["ip_address"]])

        score = 0
        if device_users >= 3:
            score += min(25, device_users * 4)
        if ip_users >= 3:
            score += min(20, ip_users * 3)

        return min(score, 35)
