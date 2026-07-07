"""路由占位模块。

当前服务端 MVP 使用 stdlib ThreadingHTTPServer，具体路由集中在
apps.server_api.backend.main.ServerRequestHandler。保留该模块是为了后续迁移到
FastAPI 或拆分 handler 时保持目录结构稳定。
"""
