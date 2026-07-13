def run_conversation(
    self,
    user_message: str,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete conversation with tool calling until completion.
    运行完整对话（含工具调用）直至结束。

    Args:
        user_message (str): The user's message/question / 用户消息
        system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
        conversation_history (List[Dict]): Previous conversation messages (optional)
        task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
        stream_callback: Optional callback invoked with each text delta during streaming.
            Used by the TTS pipeline to start audio generation before the full response.
            When None (default), API calls use the standard non-streaming path.
        persist_user_message: Optional clean user message to store in
            transcripts/history when user_message contains API-only
            synthetic prefixes.
                or queuing follow-up prefetch work.

    Returns:
        Dict: Complete conversation result with final response and message history
            完整对话结果，含 final_response 与 messages 历史
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # run_conversation() 主线模块概览 / Main Module Overview
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 【阶段 0】回合初始化 (Turn Init)
    #   stdio 防护、会话日志上下文、回退模型恢复、输入清洗、计数器重置、连接健康检查
    #
    # 【阶段 1】会话状态准备 (Session Prep)
    #   复制 history、恢复 todo、追加 user 消息、构建/复用 cached system prompt
    #
    # 【阶段 2】预检压缩 (Preflight Compression)
    #   进入主循环前，若 history 已超阈值则主动压缩（换小上下文模型时尤其重要）
    #
    # 【阶段 3】插件与记忆预取 (Plugins & Memory Prefetch)
    #   pre_llm_call 钩子注入临时上下文；memory manager prefetch（整回合复用）
    #
    # 【阶段 4】主循环 while (Main Agent Loop) — 核心
    #   4a. 预算/中断检查 → 4b. /steer 注入 → 4c. 构建 api_messages
    #       (memory 注入、reasoning 字段、prompt cache、消息 sanitize)
    #   4d. API 调用 + 内层重试循环 (streaming、错误分类、压缩、fallback)
    #   4e. 响应归一化 (Codex/Anthropic/OpenAI 适配)
    #   4f. 分支：
    #       • 有 tool_calls → 校验 → _execute_tool_calls → 可能压缩 → continue
    #       • 无 tool_calls → 空响应恢复/续写 → final_response → break
    #
    # 【阶段 5】回合收尾 (Turn Teardown)
    #   预算耗尽摘要、trajectory、资源清理、持久化、诊断日志、插件钩子、返回 result
    #
    # 退出条件: final_response 就绪 | 中断 | 预算耗尽 | 不可恢复错误
    # ═══════════════════════════════════════════════════════════════════════════

    # =======================阶段0: 回合初始化 (Turn Init)===============================

    init_conversation()

    
    
    # =======================阶段1: 会话状态准备 (Session Prep)===============================
    prepare_session()


    # =======================阶段2: 预检压缩 (Preflight Compression)===============================
    preflight_compression() 

    # =======================阶段3: 插件与记忆预取 (Plugins & Memory Prefetch)===============================
    prefetch_plugins_and_memory()

    # =======================阶段4: 主循环 while (Main Agent Loop) — 核心===============================
    # 函数的主流程
    # Main conversation loop
    # 主对话循环
    api_call_count = 0
    final_response = None
    interrupted = False
    codex_ack_continuations = 0
    length_continue_retries = 0
    truncated_tool_call_retries = 0
    truncated_response_prefix = ""
    compression_attempts = 0
    _turn_exit_reason = "unknown"  # Diagnostic: why the loop ended | 诊断：循环结束原因
    

    # Record the execution thread so interrupt()/clear_interrupt() can
    # scope the tool-level interrupt signal to THIS agent's thread only.
    # Must be set before any thread-scoped interrupt syncing.
    # 记录执行线程，使 interrupt()/clear_interrupt() 仅作用于本 agent 线程
    # 必须在线程级 interrupt 同步之前设置
    self._execution_thread_id = threading.current_thread().ident

    # Always clear stale per-thread state from a previous turn. If an
    # interrupt arrived before startup finished, preserve it and bind it
    # to this execution thread now instead of dropping it on the floor.
    # 总是清除上一回合遗留的 per-thread 状态；若启动完成前收到 interrupt，
    # 保留并绑定到本执行线程，而非丢弃
    _set_interrupt(False, self._execution_thread_id)
    if self._interrupt_requested:
        _set_interrupt(True, self._execution_thread_id)
        self._interrupt_thread_signal_pending = False
    else:
        self._interrupt_message = None
        self._interrupt_thread_signal_pending = False

    # 处理memory
    # Notify memory providers of the new turn so cadence tracking works.
    # Must happen BEFORE prefetch_all() so providers know which turn it is
    # and can gate context/dialectic refresh via contextCadence/dialecticCadence.
    # 通知 memory provider 新回合开始，使 cadence 跟踪生效
    # 必须在 prefetch_all() 之前，provider 才能按 contextCadence/dialecticCadence 控制刷新
    if self._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            self._memory_manager.on_turn_start(self._user_turn_count, _turn_msg)
        except Exception:
            pass

    # TODO： 研究这里的prefetch机制，为何可以提前获取相关信息
    # External memory provider: prefetch once before the tool loop.
    # Reuse the cached result on every iteration to avoid re-calling
    # prefetch_all() on each tool call (10 tool calls = 10x latency + cost).
    # Use original_user_message (clean input) — user_message may contain
    # injected skill content that bloats / breaks provider queries.
    # 外部 memory provider：工具循环前 prefetch 一次
    # 每轮迭代复用缓存结果，避免每次 tool call 都 prefetch（10 次 tool = 10 倍延迟/成本）
    # 使用 original_user_message（干净输入）——user_message 可能含注入 skill 内容
    _ext_prefetch_cache = ""
    if self._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    # TODO: 检查调用的轮次是否超过限制？
    while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
        # Reset per-turn checkpoint dedup so each iteration can take one snapshot
        # 重置 per-turn checkpoint 去重，使每轮迭代可拍一次快照
        self._checkpoint_mgr.new_turn()

        # Check for interrupt request (e.g., user sent new message)
        # 检查中断请求（如用户发送新消息）
        if self._interrupt_requested:
            interrupted = True
            _turn_exit_reason = "interrupted_by_user"
            if not self.quiet_mode:
                self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
            break
        
        api_call_count += 1
        self._api_call_count = api_call_count
        self._touch_activity(f"starting API call #{api_call_count}")

        # Grace call: the budget is exhausted but we gave the model one
        # more chance.  Consume the grace flag so the loop exits after
        # this iteration regardless of outcome.
        # Grace call：预算已耗尽但再给模型一次机会；消费 grace 标志后本迭代结束必退出循环
        if self._budget_grace_call:
            self._budget_grace_call = False
        elif not self.iteration_budget.consume():
            _turn_exit_reason = "budget_exhausted"
            if not self.quiet_mode:
                self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
            break

        # Fire step_callback for gateway hooks (agent:step event)
        # 触发 step_callback（gateway hooks 的 agent:step 事件）
        if self.step_callback is not None:
            try:
                prev_tools = []
                for _idx, _m in enumerate(reversed(messages)):
                    if _m.get("role") == "assistant" and _m.get("tool_calls"):
                        _fwd_start = len(messages) - _idx
                        _results_by_id = {}
                        for _tm in messages[_fwd_start:]:
                            if _tm.get("role") != "tool":
                                break
                            _tcid = _tm.get("tool_call_id")
                            if _tcid:
                                _results_by_id[_tcid] = _tm.get("content", "")
                        prev_tools = [
                            {
                                "name": tc["function"]["name"],
                                "result": _results_by_id.get(tc.get("id")),
                                "arguments": tc["function"].get("arguments"),
                            }
                            for tc in _m["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                        break
                self.step_callback(api_call_count, prev_tools)
            except Exception as _step_err:
                logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

        # Track tool-calling iterations for skill nudge.
        # Counter resets whenever skill_manage is actually used.
        # 跟踪工具调用迭代次数（skill nudge）；实际使用 skill_manage 时计数器重置
        if (self._skill_nudge_interval > 0
                and "skill_manage" in self.valid_tool_names):
            self._iters_since_skill += 1
        
        # ── Pre-API-call /steer drain ──────────────────────────────────
        # If a /steer arrived during the previous API call (while the model
        # was thinking), drain it now — before we build api_messages — so
        # the model sees the steer text on THIS iteration.  Without this,
        # steers sent during an API call only land after the NEXT tool batch,
        # which may never come if the model returns a final response.
        #
        # We scan backwards for the last tool-role message in the messages
        # list.  If found, the steer is appended there.  If not (first
        # iteration, no tools yet), the steer stays pending for the next
        # tool batch — injecting into a user message would break role
        # alternation, and there's no tool output to piggyback on.
        # ── API 调用前 /steer 排空 ──
        # 若上一 API 调用期间（模型思考时）收到 /steer，在此排空——在构建 api_messages 之前
        # 使模型在本轮迭代看到 steer 文本；否则 steer 可能只在下一批 tool 之后才生效
        #
        # 从后向前扫描 messages 中最后一条 tool 消息；找到则追加 steer
        # 否则（首轮、尚无 tool）保持 pending，等下一批 tool——注入 user 会破坏 role 交替
        _pre_api_steer = self._drain_pending_steer()
        if _pre_api_steer:
            _injected = False
            for _si in range(len(messages) - 1, -1, -1):
                _sm = messages[_si]
                if isinstance(_sm, dict) and _sm.get("role") == "tool":
                    marker = f"\n\n[USER STEER (injected mid-run, not tool output): {_pre_api_steer}]"
                    existing = _sm.get("content", "")
                    if isinstance(existing, str):
                        _sm["content"] = existing + marker
                    else:
                        # Multimodal content blocks — append text block
                        # 多模态 content blocks — 追加 text block
                        try:
                            blocks = list(existing) if existing else []
                            blocks.append({"type": "text", "text": marker})
                            _sm["content"] = blocks
                        except Exception:
                            pass
                    _injected = True
                    logger.debug(
                        "Pre-API-call steer drain: injected into tool msg at index %d",
                        _si,
                    )
                    break
            if not _injected:
                # No tool message to inject into — put it back so
                # the post-tool-execution drain picks it up later.
                # 没有可注入的 tool 消息 — 放回 pending，等 tool 执行后排空处理
                _lock = getattr(self, "_pending_steer_lock", None)
                if _lock is not None:
                    with _lock:
                        if self._pending_steer:
                            self._pending_steer = self._pending_steer + "\n" + _pre_api_steer
                        else:
                            self._pending_steer = _pre_api_steer
                else:
                    existing = getattr(self, "_pending_steer", None)
                    self._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

        # Prepare messages for API call
        # If we have an ephemeral system prompt, prepend it to the messages
        # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
        # However, providers like Moonshot AI require a separate 'reasoning_content' field
        # on assistant messages with tool_calls. We handle both cases here.
        # 准备 API 调用的 messages
        # 若有 ephemeral system prompt，prepend 到 messages
        # 注意：reasoning 通过 <think> 标签嵌入 content 供 trajectory 存储
        # 但 Moonshot AI 等 provider 要求 assistant+tool_calls 消息有独立 reasoning_content 字段——此处两者都处理
        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # Inject ephemeral context into the current turn's user message.
            # Sources: memory manager prefetch + plugin pre_llm_call hooks
            # with target="user_message" (the default).  Both are
            # API-call-time only — the original message in `messages` is
            # never mutated, so nothing leaks into session persistence.
            # 向本回合 user 消息注入 ephemeral 上下文
            # 来源：memory manager prefetch + pre_llm_call 插件（target=user_message）
            # 仅 API 调用时注入——`messages` 中原始消息不变，不会泄漏到 session 持久化
            if idx == current_turn_user_idx and msg.get("role") == "user":
                _injections = []
                if _ext_prefetch_cache:
                    _fenced = build_memory_context_block(_ext_prefetch_cache)
                    if _fenced:
                        _injections.append(_fenced)
                if _plugin_user_context:
                    _injections.append(_plugin_user_context)
                if _injections:
                    _base = api_msg.get("content", "")
                    if isinstance(_base, str):
                        api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

            # For ALL assistant messages, pass reasoning back to the API
            # This ensures multi-turn reasoning context is preserved
            # 所有 assistant 消息都把 reasoning 传回 API，保持多轮 reasoning 上下文
            if msg.get("role") == "assistant":
                reasoning_text = msg.get("reasoning")
                if reasoning_text:
                    # Add reasoning_content for API compatibility (Moonshot AI, Novita, OpenRouter)
                    # 添加 reasoning_content 以兼容 API（Moonshot AI、Novita、OpenRouter）
                    api_msg["reasoning_content"] = reasoning_text

            # Remove 'reasoning' field - it's for trajectory storage only
            # We've copied it to 'reasoning_content' for the API above
            # 移除 reasoning 字段 — 仅用于 trajectory 存储
            # 上面已复制到 reasoning_content 供 API 使用
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
            # 移除 finish_reason — 严格 API（如 Mistral）不接受
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # Strip internal thinking-prefill marker
            # 剥离内部 thinking-prefill 标记
            api_msg.pop("_thinking_prefill", None)
            # Strip Codex Responses API fields (call_id, response_item_id) for
            # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
            # Uses new dicts so the internal messages list retains the fields
            # for Codex Responses compatibility.
            # 为严格 provider（Mistral、Fireworks 等）剥离 Codex Responses API 字段（call_id、response_item_id）
            # 使用新 dict，内部 messages 列表保留字段以兼容 Codex Responses
            if self._should_sanitize_tool_calls():
                self._sanitize_tool_calls_for_strict_api(api_msg)
            # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
            # The signature field helps maintain reasoning continuity
            # 保留 reasoning_details — OpenRouter 用于多轮 reasoning 上下文
            # signature 字段有助于保持 reasoning 连续性
            api_messages.append(api_msg)

        # Build the final system message: cached prompt + ephemeral system prompt.
        # Ephemeral additions are API-call-time only (not persisted to session DB).
        # External recall context is injected into the user message, not the system
        # prompt, so the stable cache prefix remains unchanged.
        # 构建最终 system 消息：cached prompt + ephemeral system prompt
        # ephemeral 追加仅 API 调用时生效，不持久化到 session DB
        # 外部 recall 上下文注入 user 消息而非 system prompt，保持稳定 cache prefix
        effective_system = active_system_prompt or ""
        if self.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
        # NOTE: Plugin context from pre_llm_call hooks is injected into the
        # user message (see injection block above), NOT the system prompt.
        # This is intentional — system prompt modifications break the prompt
        # cache prefix.  The system prompt is reserved for Hermes internals.
        # 注意：pre_llm_call 插件上下文注入 user 消息（见上方注入块），非 system prompt
        # 有意为之 — 修改 system prompt 会破坏 prompt cache prefix；system prompt 保留给 Hermes 内部
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        # Inject ephemeral prefill messages right after the system prompt
        # but before conversation history. Same API-call-time-only pattern.
        # 在 system prompt 之后、对话 history 之前注入 ephemeral prefill 消息
        # 同样仅 API 调用时生效
        if self.prefill_messages:
            sys_offset = 1 if effective_system else 0
            for idx, pfm in enumerate(self.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Apply Anthropic prompt caching for Claude models on native
        # Anthropic, OpenRouter, and third-party Anthropic-compatible
        # gateways. Auto-detected: if ``_use_prompt_caching`` is set,
        # inject cache_control breakpoints (system + last 3 messages)
        # to reduce input token costs by ~75% on multi-turn
        # conversations. Layout is chosen per endpoint by
        # ``_anthropic_prompt_cache_policy``.
        # 为 Claude 模型应用 Anthropic prompt caching（原生 Anthropic、OpenRouter、第三方兼容网关）
        # 自动检测：若 ``_use_prompt_caching`` 为真，注入 cache_control 断点（system + 最后 3 条消息）
        # 多轮对话 input token 成本约降 75%；布局由 ``_anthropic_prompt_cache_policy`` 按端点选择
        if self._use_prompt_caching:
            api_messages = apply_anthropic_cache_control(
                api_messages,
                cache_ttl=self._cache_ttl,
                native_anthropic=self._use_native_cache_layout,
            )

        # Safety net: strip orphaned tool results / add stubs for missing
        # results before sending to the API.  Runs unconditionally — not
        # gated on context_compressor — so orphans from session loading or
        # manual message manipulation are always caught.
        # 安全网：发送 API 前剥离孤儿 tool result / 为缺失 result 添加 stub
        # 无条件运行——不依赖 context_compressor——捕获 session 加载或手动编辑产生的孤儿
        api_messages = self._sanitize_api_messages(api_messages)

        # Normalize message whitespace and tool-call JSON for consistent
        # prefix matching.  Ensures bit-perfect prefixes across turns,
        # which enables KV cache reuse on local inference servers
        # (llama.cpp, vLLM, Ollama) and improves cache hit rates for
        # cloud providers.  Operates on api_messages (the API copy) so
        # the original conversation history in `messages` is untouched.
        # 规范化消息空白和 tool-call JSON，保证 prefix 一致
        # 跨回合 bit-perfect prefix，便于本地推理（llama.cpp、vLLM、Ollama）KV cache 复用
        # 及云 provider cache 命中率；操作 api_messages（API 副本），不修改 `messages` history
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        tc["function"]["arguments"] = _repair_tool_call_arguments(
                            tc["function"]["arguments"],
                            tc["function"].get("name", "?"),
                        )
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # Proactively strip any surrogate characters before the API call.
        # Models served via Ollama (Kimi K2.5, GLM-5, Qwen) can return
        # lone surrogates (U+D800-U+DFFF) that crash json.dumps() inside
        # the OpenAI SDK. Sanitizing here prevents the 3-retry cycle.
        # API 调用前主动剥离 surrogate 字符
        # Ollama 模型（Kimi K2.5、GLM-5、Qwen）可能返回孤立 surrogate，导致 OpenAI SDK json.dumps 崩溃
        # 此处清洗可避免 3 次重试循环
        _sanitize_messages_surrogates(api_messages)

        # Calculate approximate request size for logging
        # 计算近似请求大小用于日志
        total_chars = sum(len(str(msg)) for msg in api_messages)
        approx_tokens = estimate_messages_tokens_rough(api_messages)
        
        # Thinking spinner for quiet mode (animated during API call)
        # quiet 模式下的思考 spinner（API 调用期间动画）
        thinking_spinner = None
        
        if not self.quiet_mode:
            self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
            self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
        else:
            # Animated thinking spinner in quiet mode
            # quiet 模式下的动画思考 spinner
            face = random.choice(KawaiiSpinner.get_thinking_faces())
            verb = random.choice(KawaiiSpinner.get_thinking_verbs())
            if self.thinking_callback:
                # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                # (works in both streaming and non-streaming modes)
                # CLI TUI 模式：用 prompt_toolkit 组件替代原始 spinner（流式/非流式均可用）
                self.thinking_callback(f"{face} {verb}...")
            elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                # Raw KawaiiSpinner only when no streaming consumers and the
                # spinner output has a safe sink.
                # 仅当无 streaming consumer 且 spinner 有安全输出 sink 时使用原始 KawaiiSpinner
                spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                thinking_spinner.start()
        
        # Log request details if verbose
        # verbose 模式下记录请求详情
        if self.verbose_logging:
            logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
            logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
        
        api_start_time = time.time()
        retry_count = 0
        max_retries = 3
        primary_recovery_attempted = False
        max_compression_attempts = 3
        codex_auth_retry_attempted=False
        anthropic_auth_retry_attempted=False
        nous_auth_retry_attempted=False
        thinking_sig_retry_attempted = False
        has_retried_429 = False
        restart_with_compressed_messages = False
        restart_with_length_continuation = False

        finish_reason = "stop"
        response = None  # Guard against UnboundLocalError if all retries fail | 防止所有重试失败时 UnboundLocalError
        api_kwargs = None  # Guard against UnboundLocalError in except handler | 防止 except 处理器中 UnboundLocalError

        while retry_count < max_retries:
            # ── Nous Portal rate limit guard ──────────────────────
            # If another session already recorded that Nous is rate-
            # limited, skip the API call entirely.  Each attempt
            # (including SDK-level retries) counts against RPH and
            # deepens the rate limit hole.
            # ── Nous Portal 限速守卫 ──
            # 若其他 session 已记录 Nous 被限速，完全跳过 API 调用
            # 每次尝试（含 SDK 级重试）都计入 RPH，加深限速
            if self.provider == "nous":
                try:
                    from agent.nous_rate_guard import (
                        nous_rate_limit_remaining,
                        format_remaining as _fmt_nous_remaining,
                    )
                    _nous_remaining = nous_rate_limit_remaining()
                    if _nous_remaining is not None and _nous_remaining > 0:
                        _nous_msg = (
                            f"Nous Portal rate limit active — "
                            f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                        )
                        self._vprint(
                            f"{self.log_prefix}⏳ {_nous_msg} Trying fallback...",
                            force=True,
                        )
                        self._emit_status(f"⏳ {_nous_msg}")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        # No fallback available — return with clear message
                        # 无 fallback 可用 — 返回明确消息
                        self._persist_session(messages, conversation_history)
                        return {
                            "final_response": (
                                f"⏳ {_nous_msg}\n\n"
                                "No fallback provider available. "
                                "Try again after the reset, or add a "
                                "fallback provider in config.yaml."
                            ),
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nous_msg,
                        }
                except ImportError:
                    pass
                except Exception:
                    pass  # Never let rate guard break the agent loop | 绝不让 rate guard 破坏 agent loop

            try:
                self._reset_stream_delivery_tracking()
                api_kwargs = self._build_api_kwargs(api_messages)
                if self._force_ascii_payload:
                    _sanitize_structure_non_ascii(api_kwargs)
                if self.api_mode == "codex_responses":
                    api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "pre_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        message_count=len(api_messages),
                        tool_count=len(self.tools or []),
                        approx_input_tokens=approx_tokens,
                        request_char_count=total_chars,
                        max_tokens=self.max_tokens,
                    )
                except Exception:
                    pass

                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    self._dump_api_request_debug(api_kwargs, reason="preflight")

                # Always prefer the streaming path — even without stream
                # consumers.  Streaming gives us fine-grained health
                # checking (90s stale-stream detection, 60s read timeout)
                # that the non-streaming path lacks.  Without this,
                # subagents and other quiet-mode callers can hang
                # indefinitely when the provider keeps the connection
                # alive with SSE pings but never delivers a response.
                # The streaming path is a no-op for callbacks when no
                # consumers are registered, and falls back to non-
                # streaming automatically if the provider doesn't
                # support it.
                # 始终优先 streaming 路径——即使没有 stream consumer
                # streaming 提供细粒度健康检查（90s stale-stream、60s read timeout），非 streaming 没有
                # 否则 subagent 等 quiet 调用可能在 provider 只发 SSE ping 无响应时无限挂起
                # 无 consumer 时 streaming 对 callback 是 no-op；provider 不支持时自动回退非 streaming
                def _stop_spinner():
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                _use_streaming = True
                # Provider signaled "stream not supported" on a previous
                # attempt — switch to non-streaming for the rest of this
                # session instead of re-failing every retry.
                # provider 此前 signaled "stream not supported" — 本会话剩余部分改用非 streaming，避免每次重试都失败
                if getattr(self, "_disable_streaming", False):
                    _use_streaming = False
                elif not self._has_stream_consumers():
                    # No display/TTS consumer. Still prefer streaming for
                    # health checking, but skip for Mock clients in tests
                    # (mocks return SimpleNamespace, not stream iterators).
                    # 无 display/TTS consumer；仍优先 streaming 做健康检查，但测试 Mock client 跳过
                    #（mock 返回 SimpleNamespace，非 stream iterator）
                    from unittest.mock import Mock
                    if isinstance(getattr(self, "client", None), Mock):
                        _use_streaming = False

                # 调用大模型
                if _use_streaming:
                    response = self._interruptible_streaming_api_call(
                        api_kwargs, on_first_delta=_stop_spinner
                    )
                else:
                    response = self._interruptible_api_call(api_kwargs)
                
                api_duration = time.time() - api_start_time
                
                # Stop thinking spinner silently -- the response box or tool
                # execution messages that follow are more informative.
                # 静默停止思考 spinner — 后续 response box 或 tool 执行消息更有信息量
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if self.thinking_callback:
                    self.thinking_callback("")
                
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                
                if self.verbose_logging:
                    # Log response with provider info if available
                    # 若可用，记录带 provider 信息的响应
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                
                # Validate response shape before proceeding
                # 继续处理前先校验响应结构
                response_invalid = False
                error_details = []
                if self.api_mode == "codex_responses":
                    output_items = getattr(response, "output", None) if response is not None else None
                    if response is None:
                        response_invalid = True
                        error_details.append("response is None")
                    elif not isinstance(output_items, list):
                        response_invalid = True
                        error_details.append("response.output is not a list")
                    elif not output_items:
                        # Stream backfill may have failed, but
                        # _normalize_codex_response can still recover
                        # from response.output_text. Only mark invalid
                        # when that fallback is also absent.
                        # stream backfill 可能失败，但 _normalize_codex_response 仍可从 response.output_text 恢复
                        # 仅当该 fallback 也不存在时才标记 invalid
                        _out_text = getattr(response, "output_text", None)
                        _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                        if _out_text_stripped:
                            logger.debug(
                                "Codex response.output is empty but output_text is present "
                                "(%d chars); deferring to normalization.",
                                len(_out_text_stripped),
                            )
                        else:
                            _resp_status = getattr(response, "status", None)
                            _resp_incomplete = getattr(response, "incomplete_details", None)
                            logger.warning(
                                "Codex response.output is empty after stream backfill "
                                "(status=%s, incomplete_details=%s, model=%s). %s",
                                _resp_status, _resp_incomplete,
                                getattr(response, "model", None),
                                f"api_mode={self.api_mode} provider={self.provider}",
                            )
                            response_invalid = True
                            error_details.append("response.output is empty")
                elif self.api_mode == "anthropic_messages":
                    content_blocks = getattr(response, "content", None) if response is not None else None
                    if response is None:
                        response_invalid = True
                        error_details.append("response is None")
                    elif not isinstance(content_blocks, list):
                        response_invalid = True
                        error_details.append("response.content is not a list")
                    elif not content_blocks:
                        response_invalid = True
                        error_details.append("response.content is empty")
                else:
                    if response is None or not hasattr(response, 'choices') or response.choices is None or not response.choices:
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")

                if response_invalid:
                    # Stop spinner before printing error messages
                    # 打印错误消息前先停止 spinner
                    if thinking_spinner:
                        thinking_spinner.stop("(´;ω;`) oops, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    
                    # Invalid response — could be rate limiting, provider timeout,
                    # upstream server error, or malformed response.
                    # 无效响应 — 可能是限速、provider 超时、上游错误或畸形响应
                    retry_count += 1
                    
                    # Eager fallback: empty/malformed responses are a common
                    # rate-limit symptom.  Switch to fallback immediately
                    # rather than retrying with extended backoff.
                    # 积极 fallback：空/畸形响应常见于限速；立即切换 fallback 而非长 backoff 重试
                    if self._fallback_index < len(self._fallback_chain):
                        self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                    if self._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue

                    # Check for error field in response (some providers include this)
                    # 检查响应中的 error 字段（部分 provider 会包含）
                    error_msg = "Unknown"
                    provider_name = "Unknown"
                    if response and hasattr(response, 'error') and response.error:
                        error_msg = str(response.error)
                        # Try to extract provider from error metadata
                        # 尝试从 error metadata 提取 provider
                        if hasattr(response.error, 'metadata') and response.error.metadata:
                            provider_name = response.error.metadata.get('provider_name', 'Unknown')
                    elif response and hasattr(response, 'message') and response.message:
                        error_msg = str(response.message)
                    
                    # Try to get provider from model field (OpenRouter often returns actual model used)
                    # 尝试从 model 字段获取 provider（OpenRouter 常返回实际使用的模型）
                    if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                        provider_name = f"model={response.model}"
                    
                    # Check for x-openrouter-provider or similar metadata
                    # 检查 x-openrouter-provider 或类似 metadata
                    if provider_name == "Unknown" and response:
                        # Log all response attributes for debugging
                        # 记录所有响应属性用于调试
                        resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                        if self.verbose_logging:
                            logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                    
                    # Extract error code from response for contextual diagnostics
                    # 从响应提取 error code，用于上下文诊断
                    _resp_error_code = None
                    if response and hasattr(response, 'error') and response.error:
                        _code_raw = getattr(response.error, 'code', None)
                        if _code_raw is None and isinstance(response.error, dict):
                            _code_raw = response.error.get('code')
                        if _code_raw is not None:
                            try:
                                _resp_error_code = int(_code_raw)
                            except (TypeError, ValueError):
                                pass

                    # Build a human-readable failure hint from the error code
                    # and response time, instead of always assuming rate limiting.
                    # 根据 error code 和响应时间构建可读失败提示，而非一律假设限速
                    if _resp_error_code == 524:
                        _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                    elif _resp_error_code == 504:
                        _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                    elif _resp_error_code == 429:
                        _failure_hint = f"rate limited by upstream provider (429)"
                    elif _resp_error_code in (500, 502):
                        _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                    elif _resp_error_code in (503, 529):
                        _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                    elif _resp_error_code is not None:
                        _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                    elif api_duration < 10:
                        _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                    elif api_duration > 60:
                        _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                    else:
                        _failure_hint = f"response time {api_duration:.1f}s"

                    self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                    self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                    cleaned_provider_error = self._clean_error_message(error_msg)
                    self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  {_failure_hint}", force=True)
                    
                    if retry_count >= max_retries:
                        # Try fallback before giving up
                        # 放弃前先尝试 fallback
                        self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                        self._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                            "failed": True  # Mark as failure for filtering
                        }
                    
                    # Backoff before retry — jittered exponential: 5s base, 120s cap
                    # 重试前 backoff — 抖动指数退避：5s 基数，120s 上限
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...", force=True)
                    logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                    
                    # Sleep in small increments to stay responsive to interrupts
                    # 小步 sleep 以保持对 interrupt 的响应
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)
                        # Touch activity every ~30s so the gateway's inactivity
                        # monitor knows we're alive during backoff waits.
                        # 约每 30s touch activity，让 gateway 不活动监控知道 backoff 期间仍存活
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s | 约 30 秒
                            self._touch_activity(
                                f"retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
                    continue  # Retry the API call | 重试 API 调用

                # Check finish_reason before proceeding
                # 继续前先检查 finish_reason
                if self.api_mode == "codex_responses":
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                elif self.api_mode == "anthropic_messages":
                    stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                    finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                else:
                    finish_reason = response.choices[0].finish_reason
                    assistant_message = response.choices[0].message
                    if self._should_treat_stop_as_truncated(
                        finish_reason,
                        assistant_message,
                        messages,
                    ):
                        self._vprint(
                            f"{self.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                            force=True,
                        )
                        finish_reason = "length"

                if finish_reason == "length":
                    self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                    # Normalize the truncated response to a single OpenAI-style
                    # message shape so text-continuation and tool-call retry
                    # work uniformly across chat_completions, bedrock_converse,
                    # and anthropic_messages.  For Anthropic we use the same
                    # adapter the agent loop already relies on so the rebuilt
                    # interim assistant message is byte-identical to what
                    # would have been appended in the non-truncated path.
                    # 将截断响应规范化为统一 OpenAI 风格 message，使文本续写和 tool-call 重试
                    # 在 chat_completions、bedrock_converse、anthropic_messages 上一致工作
                    # Anthropic 使用 agent loop 已有 adapter，重建的 interim assistant 与非截断路径 byte-identical
                    _trunc_msg = None
                    if self.api_mode in ("chat_completions", "bedrock_converse"):
                        _trunc_msg = response.choices[0].message if (hasattr(response, "choices") and response.choices) else None
                    elif self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import normalize_anthropic_response
                        _trunc_msg, _ = normalize_anthropic_response(
                            response, strip_tool_prefix=self._is_anthropic_oauth
                        )

                    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                    # ── Detect thinking-budget exhaustion ──────────────
                    # When the model spends ALL output tokens on reasoning
                    # and has none left for the response, continuation
                    # retries are pointless.  Detect this early and give a
                    # targeted error instead of wasting 3 API calls.
                    # A response is "thinking exhausted" only when the model
                    # actually produced reasoning blocks but no visible text after
                    # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
                    # NVIDIA Build, minimax) may return content=None or an empty
                    # string for unrelated reasons — treat those as normal
                    # truncations that deserve continuation retries, not as
                    # thinking-budget exhaustion.
                    # ── 检测 thinking 预算耗尽 ──
                    # 模型把所有 output token 用于 reasoning 而无剩余给响应时，续写重试无意义
                    # 尽早检测并给出针对性错误，避免浪费 3 次 API 调用
                    # "thinking exhausted" 仅当模型确实产生 reasoning 块但之后无可见文本
                    # 不使用 <think> 的模型（如 NVIDIA Build 上 GLM-4.7、minimax）可能 content=None
                    # 或空字符串出于其他原因 — 视为正常截断应续写，非 thinking 预算耗尽
                    _has_think_tags = bool(
                        _trunc_content and re.search(
                            r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                            _trunc_content,
                            re.IGNORECASE,
                        )
                    )
                    _thinking_exhausted = (
                        not _trunc_has_tool_calls
                        and _has_think_tags
                        and (
                            (_trunc_content is not None and not self._has_content_after_think_block(_trunc_content))
                            or _trunc_content is None
                        )
                    )

                    if _thinking_exhausted:
                        _exhaust_error = (
                            "Model used all output tokens on reasoning with none left "
                            "for the response. Try lowering reasoning effort or "
                            "increasing max_tokens."
                        )
                        self._vprint(
                            f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                            f"no visible response was produced.",
                            force=True,
                        )
                        # Return a user-friendly message as the response so
                        # CLI (response box) and gateway (chat message) both
                        # display it naturally instead of a suppressed error.
                        # 返回用户友好消息作为响应，CLI（response box）和 gateway（聊天消息）自然展示
                        # 而非 suppressed error
                        _exhaust_response = (
                            "⚠️ **Thinking Budget Exhausted**\n\n"
                            "The model used all its output tokens on reasoning "
                            "and had none left for the actual response.\n\n"
                            "To fix this:\n"
                            "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                            "→ Or switch to a larger/non-reasoning model with `/model`"
                        )
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        return {
                            "final_response": _exhaust_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": _exhaust_error,
                        }

                    if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
                        assistant_message = _trunc_msg
                        if assistant_message is not None and not _trunc_has_tool_calls:
                            length_continue_retries += 1
                            interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(interim_msg)
                            if assistant_message.content:
                                truncated_response_prefix += assistant_message.content

                            if length_continue_retries < 3:
                                self._vprint(
                                    f"{self.log_prefix}↻ Requesting continuation "
                                    f"({length_continue_retries}/3)..."
                                )
                                continue_msg = {
                                    "role": "user",
                                    "content": (
                                        "[System: Your previous response was truncated by the output "
                                        "length limit. Continue exactly where you left off. Do not "
                                        "restart or repeat prior text. Finish the answer directly.]"
                                    ),
                                }
                                messages.append(continue_msg)
                                self._session_messages = messages
                                self._save_session_log(messages)
                                restart_with_length_continuation = True
                                break

                            partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": partial_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response remained truncated after 3 continuation attempts",
                            }

                    if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
                        assistant_message = _trunc_msg
                        if assistant_message is not None and _trunc_has_tool_calls:
                            if truncated_tool_call_retries < 1:
                                truncated_tool_call_retries += 1
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Truncated tool call detected — retrying API call...",
                                    force=True,
                                )
                                # Don't append the broken response to messages;
                                # just re-run the same API call from the current
                                # message state, giving the model another chance.
                                # 不把 broken response 追加到 messages
                                # 从当前 message 状态重跑同一 API 调用，再给模型一次机会
                                continue
                            self._vprint(
                                f"{self.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                force=True,
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                    # If we have prior messages, roll back to last complete state
                    # 若有 prior messages，回滚到最后完整状态
                    if len(messages) > 1:
                        self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)

                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit"
                        }
                    else:
                        # First message was truncated - mark as failed
                        # 首条消息被截断 — 标记失败
                        self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                        self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": "First response truncated due to output length limit"
                        }
                
                # Track actual token usage from response for context management
                # 从响应跟踪实际 token 用量，用于上下文管理
                if hasattr(response, 'usage') and response.usage:
                    canonical_usage = normalize_usage(
                        response.usage,
                        provider=self.provider,
                        api_mode=self.api_mode,
                    )
                    prompt_tokens = canonical_usage.prompt_tokens
                    completion_tokens = canonical_usage.output_tokens
                    total_tokens = canonical_usage.total_tokens
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                    self.context_compressor.update_from_response(usage_dict)

                    # Cache discovered context length after successful call.
                    # Only persist limits confirmed by the provider (parsed
                    # from the error message), not guessed probe tiers.
                    # 成功调用后缓存发现的 context length
                    # 仅持久化 provider 确认的限制（从 error 解析），非猜测的 probe tier
                    if getattr(self.context_compressor, "_context_probed", False):
                        ctx = self.context_compressor.context_length
                        if getattr(self.context_compressor, "_context_probe_persistable", False):
                            save_context_length(self.model, self.base_url, ctx)
                            self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                        self.context_compressor._context_probed = False
                        self.context_compressor._context_probe_persistable = False

                    self.session_prompt_tokens += prompt_tokens
                    self.session_completion_tokens += completion_tokens
                    self.session_total_tokens += total_tokens
                    self.session_api_calls += 1
                    self.session_input_tokens += canonical_usage.input_tokens
                    self.session_output_tokens += canonical_usage.output_tokens
                    self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                    self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                    self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                    # Log API call details for debugging/observability
                    # 记录 API 调用详情，便于调试/可观测性
                    _cache_pct = ""
                    if canonical_usage.cache_read_tokens and prompt_tokens:
                        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                    logger.info(
                        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                        self.session_api_calls, self.model, self.provider or "unknown",
                        prompt_tokens, completion_tokens, total_tokens,
                        api_duration, _cache_pct,
                    )

                    cost_result = estimate_usage_cost(
                        self.model,
                        canonical_usage,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_key=getattr(self, "api_key", ""),
                    )
                    if cost_result.amount_usd is not None:
                        self.session_estimated_cost_usd += float(cost_result.amount_usd)
                    self.session_cost_status = cost_result.status
                    self.session_cost_source = cost_result.source

                    # Persist token counts to session DB for /insights.
                    # Do this for every platform with a session_id so non-CLI
                    # sessions (gateway, cron, delegated runs) cannot lose
                    # token/accounting data if a higher-level persistence path
                    # is skipped or fails. Gateway/session-store writes use
                    # absolute totals, so they safely overwrite these per-call
                    # deltas instead of double-counting them.
                    # 持久化 token 计数到 session DB（/insights）
                    # 所有有 session_id 的平台都写，避免 gateway/cron/delegate 丢失 token 统计
                    # gateway/session-store 用绝对总量覆盖 per-call delta，不会重复计数
                    if self._session_db and self.session_id:
                        try:
                            self._session_db.update_token_counts(
                                self.session_id,
                                input_tokens=canonical_usage.input_tokens,
                                output_tokens=canonical_usage.output_tokens,
                                cache_read_tokens=canonical_usage.cache_read_tokens,
                                cache_write_tokens=canonical_usage.cache_write_tokens,
                                reasoning_tokens=canonical_usage.reasoning_tokens,
                                estimated_cost_usd=float(cost_result.amount_usd)
                                if cost_result.amount_usd is not None else None,
                                cost_status=cost_result.status,
                                cost_source=cost_result.source,
                                billing_provider=self.provider,
                                billing_base_url=self.base_url,
                                billing_mode="subscription_included"
                                if cost_result.status == "included" else None,
                                model=self.model,
                            )
                        except Exception:
                            pass  # never block the agent loop | 绝不妨碍 agent loop
                    
                    if self.verbose_logging:
                        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                    
                    # Log cache hit stats when prompt caching is active
                    # prompt caching 激活时记录 cache 命中统计
                    if self._use_prompt_caching:
                        if self.api_mode == "anthropic_messages":
                            # Anthropic uses cache_read_input_tokens / cache_creation_input_tokens
                            # Anthropic 使用 cache_read_input_tokens / cache_creation_input_tokens
                            cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                            written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                        else:
                            # OpenRouter uses prompt_tokens_details.cached_tokens
                            # OpenRouter 使用 prompt_tokens_details.cached_tokens
                            details = getattr(response.usage, 'prompt_tokens_details', None)
                            cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                            written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                        prompt = usage_dict["prompt_tokens"]
                        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")
                
                has_retried_429 = False  # Reset on success
                # Clear Nous rate limit state on successful request —
                # proves the limit has reset and other sessions can
                # resume hitting Nous.
                # 成功请求后清除 Nous 限速状态 — 证明限制已重置，其他 session 可继续用 Nous
                if self.provider == "nous":
                    try:
                        from agent.nous_rate_guard import clear_nous_rate_limit
                        clear_nous_rate_limit()
                    except Exception:
                        pass
                self._touch_activity(f"API call #{api_call_count} completed")
                break  # Success, exit retry loop | 成功，退出重试循环

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if self.thinking_callback:
                    self.thinking_callback("")
                api_elapsed = time.time() - api_start_time
                self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                self._persist_session(messages, conversation_history)
                interrupted = True
                final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                break

            except Exception as api_error:
                # Stop spinner before printing error messages
                # 打印错误消息前先停止 spinner
                if thinking_spinner:
                    thinking_spinner.stop("(╥_╥) error, retrying...")
                    thinking_spinner = None
                if self.thinking_callback:
                    self.thinking_callback("")

                # -----------------------------------------------------------
                # UnicodeEncodeError recovery.  Two common causes:
                #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
                #      (Google Docs, rich-text editors) — sanitize and retry.
                #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
                #      (e.g. Chromebooks) — any non-ASCII character fails.
                #      Detect via the error message mentioning 'ascii' codec.
                # We sanitize messages in-place and may retry twice:
                # first to strip surrogates, then once more for pure
                # ASCII-only locale sanitization if needed.
                # -----------------------------------------------------------
                # -----------------------------------------------------------
                # UnicodeEncodeError 恢复。两种常见原因：
                #   1. 剪贴板粘贴的孤立 surrogate (U+D800..U+DFFF) — 清洗后重试
                #   2. LANG=C 或非 UTF-8 locale 的 ASCII codec（如 Chromebook）— 任何非 ASCII 都会失败
                #      通过 error 消息含 'ascii' codec 检测
                # 就地清洗 messages，最多重试两次：先剥离 surrogate，再纯 ASCII locale 清洗
                # -----------------------------------------------------------
                if isinstance(api_error, UnicodeEncodeError) and getattr(self, '_unicode_sanitization_passes', 0) < 2:
                    _err_str = str(api_error).lower()
                    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                    # Detect surrogate errors — utf-8 codec refusing to
                    # encode U+D800..U+DFFF.  The error text is:
                    #   "'utf-8' codec can't encode characters in position
                    #    N-M: surrogates not allowed"
                    # 检测 surrogate 错误 — utf-8 codec 拒绝编码 U+D800..U+DFFF
                    # error 文本示例："'utf-8' codec can't encode characters in position N-M: surrogates not allowed"
                    _is_surrogate_error = (
                        "surrogate" in _err_str
                        or ("'utf-8'" in _err_str and not _is_ascii_codec)
                    )
                    # Sanitize surrogates from both the canonical `messages`
                    # list AND `api_messages` (the API-copy, which may carry
                    # `reasoning_content`/`reasoning_details` transformed
                    # from `reasoning` — fields the canonical list doesn't
                    # have directly).  Also clean `api_kwargs` if built and
                    # `prefill_messages` if present.  Mirrors the ASCII
                    # codec recovery below.
                    # 从 canonical `messages` 和 `api_messages`（API 副本，可能含 reasoning_content/reasoning_details）
                    # 清洗 surrogate；若已构建则清洗 `api_kwargs` 和 `prefill_messages`；镜像下方 ASCII codec 恢复
                    _surrogates_found = _sanitize_messages_surrogates(messages)
                    if isinstance(api_messages, list):
                        if _sanitize_messages_surrogates(api_messages):
                            _surrogates_found = True
                    if isinstance(api_kwargs, dict):
                        if _sanitize_structure_surrogates(api_kwargs):
                            _surrogates_found = True
                    if isinstance(getattr(self, "prefill_messages", None), list):
                        if _sanitize_messages_surrogates(self.prefill_messages):
                            _surrogates_found = True
                    # Gate the retry on the error type, not on whether we
                    # found anything — _force_ascii_payload / the extended
                    # surrogate walker above cover all known paths, but a
                    # new transformed field could still slip through.  If
                    # the error was a surrogate encode failure, always let
                    # the retry run; the proactive sanitizer at line ~8781
                    # runs again on the next iteration.  Bounded by
                    # _unicode_sanitization_passes < 2 (outer guard).
                    # 重试门控基于 error 类型而非是否找到 surrogate — _force_ascii_payload / 扩展 walker 覆盖已知路径
                    # 新 transform 字段可能仍漏网；surrogate encode 失败时总是允许重试
                    # 下次迭代 proactive sanitizer（约 line ~8781）会再跑；由 _unicode_sanitization_passes < 2 限制
                    if _surrogates_found or _is_surrogate_error:
                        self._unicode_sanitization_passes += 1
                        if _surrogates_found:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                force=True,
                            )
                        else:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Surrogate encoding error — retrying after full-payload sanitization...",
                                force=True,
                            )
                        continue
                    if _is_ascii_codec:
                        self._force_ascii_payload = True
                        # ASCII codec: the system encoding can't handle
                        # non-ASCII characters at all. Sanitize all
                        # non-ASCII content from messages/tool schemas and retry.
                        # Sanitize both the canonical `messages` list and
                        # `api_messages` (the API-copy built before the retry
                        # loop, which may contain extra fields like
                        # reasoning_content that are not in `messages`).
                        # ASCII codec：系统编码完全无法处理非 ASCII — 清洗 messages/tool schema 后重试
                        # 同时清洗 canonical `messages` 和 `api_messages`（重试循环前构建，可能含 messages 中没有的 reasoning_content）
                        _messages_sanitized = _sanitize_messages_non_ascii(messages)
                        if isinstance(api_messages, list):
                            _sanitize_messages_non_ascii(api_messages)
                        # Also sanitize the last api_kwargs if already built,
                        # so a leftover non-ASCII value in a transformed field
                        # (e.g. extra_body, reasoning_content) doesn't survive
                        # into the next attempt via _build_api_kwargs cache paths.
                        # 若 api_kwargs 已构建也清洗，避免 transformed 字段（extra_body、reasoning_content）
                        # 通过 _build_api_kwargs 缓存路径带入下次尝试
                        if isinstance(api_kwargs, dict):
                            _sanitize_structure_non_ascii(api_kwargs)
                        _prefill_sanitized = False
                        if isinstance(getattr(self, "prefill_messages", None), list):
                            _prefill_sanitized = _sanitize_messages_non_ascii(self.prefill_messages)

                        _tools_sanitized = False
                        if isinstance(getattr(self, "tools", None), list):
                            _tools_sanitized = _sanitize_tools_non_ascii(self.tools)

                        _system_sanitized = False
                        if isinstance(active_system_prompt, str):
                            _sanitized_system = _strip_non_ascii(active_system_prompt)
                            if _sanitized_system != active_system_prompt:
                                active_system_prompt = _sanitized_system
                                self._cached_system_prompt = _sanitized_system
                                _system_sanitized = True
                        if isinstance(getattr(self, "ephemeral_system_prompt", None), str):
                            _sanitized_ephemeral = _strip_non_ascii(self.ephemeral_system_prompt)
                            if _sanitized_ephemeral != self.ephemeral_system_prompt:
                                self.ephemeral_system_prompt = _sanitized_ephemeral
                                _system_sanitized = True

                        _headers_sanitized = False
                        _default_headers = (
                            self._client_kwargs.get("default_headers")
                            if isinstance(getattr(self, "_client_kwargs", None), dict)
                            else None
                        )
                        if isinstance(_default_headers, dict):
                            _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                        # Sanitize the API key — non-ASCII characters in
                        # credentials (e.g. ʋ instead of v from a bad
                        # copy-paste) cause httpx to fail when encoding
                        # the Authorization header as ASCII.  This is the
                        # most common cause of persistent UnicodeEncodeError
                        # that survives message/tool sanitization (#6843).
                        # 清洗 API key — 凭证中非 ASCII（如错误粘贴的 ʋ 代替 v）导致 httpx 编码 Authorization 头失败
                        # 这是 message/tool 清洗后仍 persistent UnicodeEncodeError 的最常见原因 (#6843)
                        _credential_sanitized = False
                        _raw_key = getattr(self, "api_key", None) or ""
                        if _raw_key:
                            _clean_key = _strip_non_ascii(_raw_key)
                            if _clean_key != _raw_key:
                                self.api_key = _clean_key
                                if isinstance(getattr(self, "_client_kwargs", None), dict):
                                    self._client_kwargs["api_key"] = _clean_key
                                # Also update the live client — it holds its
                                # own copy of api_key which auth_headers reads
                                # dynamically on every request.
                                # 同时更新 live client — 它持有 api_key 副本，auth_headers 每次请求动态读取
                                if getattr(self, "client", None) is not None and hasattr(self.client, "api_key"):
                                    self.client.api_key = _clean_key
                                _credential_sanitized = True
                                self._vprint(
                                    f"{self.log_prefix}⚠️  API key contained non-ASCII characters "
                                    f"(bad copy-paste?) — stripped them. If auth fails, "
                                    f"re-copy the key from your provider's dashboard.",
                                    force=True,
                                )

                        # Always retry on ASCII codec detection —
                        # _force_ascii_payload guarantees the full
                        # api_kwargs payload is sanitized on the
                        # next iteration (line ~8475).  Even when
                        # per-component checks above find nothing
                        # (e.g. non-ASCII only in api_messages'
                        # reasoning_content), the flag catches it.
                        # Bounded by _unicode_sanitization_passes < 2.
                        # 检测到 ASCII codec 时总是重试 — _force_ascii_payload 保证下次迭代完整 api_kwargs 被清洗
                        # 即使上方 per-component 检查未发现（如非 ASCII 仅在 api_messages reasoning_content）
                        # flag 也能捕获；由 _unicode_sanitization_passes < 2 限制
                        self._unicode_sanitization_passes += 1
                        _any_sanitized = (
                            _messages_sanitized
                            or _prefill_sanitized
                            or _tools_sanitized
                            or _system_sanitized
                            or _headers_sanitized
                            or _credential_sanitized
                        )
                        if _any_sanitized:
                            self._vprint(
                                f"{self.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                force=True,
                            )
                        else:
                            self._vprint(
                                f"{self.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                force=True,
                            )
                        continue

                status_code = getattr(api_error, "status_code", None)
                error_context = self._extract_api_error_context(api_error)

                # ── Classify the error for structured recovery decisions ──
                # ── 分类 error 以做结构化恢复决策 ──
                _compressor = getattr(self, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                classified = classify_api_error(
                    api_error,
                    provider=getattr(self, "provider", "") or "",
                    model=getattr(self, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )

                recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                    status_code=status_code,
                    has_retried_429=has_retried_429,
                    classified_reason=classified.reason,
                    error_context=error_context,
                )
                if recovered_with_pool:
                    continue
                if (
                    self.api_mode == "codex_responses"
                    and self.provider == "openai-codex"
                    and status_code == 401
                    and not codex_auth_retry_attempted
                ):
                    codex_auth_retry_attempted = True
                    if self._try_refresh_codex_client_credentials(force=True):
                        self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                        continue
                if (
                    self.api_mode == "chat_completions"
                    and self.provider == "nous"
                    and status_code == 401
                    and not nous_auth_retry_attempted
                ):
                    nous_auth_retry_attempted = True
                    if self._try_refresh_nous_client_credentials(force=True):
                        print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                        continue
                if (
                    self.api_mode == "anthropic_messages"
                    and status_code == 401
                    and hasattr(self, '_anthropic_api_key')
                    and not anthropic_auth_retry_attempted
                ):
                    anthropic_auth_retry_attempted = True
                    from agent.anthropic_adapter import _is_oauth_token
                    if self._try_refresh_anthropic_client_credentials():
                        print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info
                    # 凭证刷新无效 — 显示诊断信息
                    key = self._anthropic_api_key
                    auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                    print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
                    print(f"{self.log_prefix}   Auth method: {auth_method}")
                    print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
                    print(f"{self.log_prefix}   Troubleshooting:")
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                    print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                    print(f"{self.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                    print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                    print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                    print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                # ── Thinking block signature recovery ─────────────────
                # Anthropic signs thinking blocks against the full turn
                # content.  Any upstream mutation (context compression,
                # session truncation, message merging) invalidates the
                # signature → HTTP 400.  Recovery: strip reasoning_details
                # from all messages so the next retry sends no thinking
                # blocks at all.  One-shot — don't retry infinitely.
                # ── Thinking block 签名恢复 ──
                # Anthropic 对完整 turn content 签名 thinking block；上游变更（压缩、截断、合并）使签名失效 → HTTP 400
                # 恢复：剥离所有 messages 的 reasoning_details，下次重试不发送 thinking block；一次性，不无限重试
                if (
                    classified.reason == FailoverReason.thinking_signature
                    and not thinking_sig_retry_attempted
                ):
                    thinking_sig_retry_attempted = True
                    for _m in messages:
                        if isinstance(_m, dict):
                            _m.pop("reasoning_details", None)
                    self._vprint(
                        f"{self.log_prefix}⚠️  Thinking block signature invalid — "
                        f"stripped all thinking blocks, retrying...",
                        force=True,
                    )
                    logging.warning(
                        "%sThinking block signature recovery: stripped "
                        "reasoning_details from %d messages",
                        self.log_prefix, len(messages),
                    )
                    continue

                retry_count += 1
                elapsed_time = time.time() - api_start_time
                self._touch_activity(
                    f"API error recovery (attempt {retry_count}/{max_retries})"
                )
                
                error_type = type(api_error).__name__
                error_msg = str(api_error).lower()
                _error_summary = self._summarize_api_error(api_error)
                logger.warning(
                    "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                    retry_count,
                    max_retries,
                    error_type,
                    self._client_log_context(),
                    _error_summary,
                )

                _provider = getattr(self, "provider", "unknown")
                _base = getattr(self, "base_url", "unknown")
                _model = getattr(self, "model", "unknown")
                _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
                if status_code and status_code < 500:
                    _err_body = getattr(api_error, "body", None)
                    _err_body_str = str(_err_body)[:300] if _err_body else None
                    if _err_body_str:
                        self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
                self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                # Actionable hint for OpenRouter "no tool endpoints" error.
                # This fires regardless of whether fallback succeeds — the
                # user needs to know WHY their model failed so they can fix
                # their provider routing, not just silently fall back.
                # OpenRouter "no tool endpoints" 错误的可操作提示
                # 无论 fallback 是否成功都触发 — 用户需知道失败原因以修复 provider 路由，而非静默 fallback
                if (
                    self._is_openrouter_url()
                    and "support tool use" in error_msg
                ):
                    self._vprint(
                        f"{self.log_prefix}   💡 No OpenRouter providers for {_model} support tool calling with your current settings.",
                        force=True,
                    )
                    if self.providers_allowed:
                        self._vprint(
                            f"{self.log_prefix}      Your provider_routing.only restriction is filtering out tool-capable providers.",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}      Try removing the restriction or adding providers that support tools for this model.",
                            force=True,
                        )
                    self._vprint(
                        f"{self.log_prefix}      Check which providers support tools: https://openrouter.ai/models/{_model}",
                        force=True,
                    )

                # Check for interrupt before deciding to retry
                # 决定重试前先检查 interrupt
                if self._interrupt_requested:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    self._persist_session(messages, conversation_history)
                    self.clear_interrupt()
                    return {
                        "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }
                
                # Check for 413 payload-too-large BEFORE generic 4xx handler.
                # A 413 is a payload-size error — the correct response is to
                # compress history and retry, not abort immediately.
                # 在通用 4xx 处理前先检查 413 payload-too-large
                # 413 是 payload 过大 — 正确响应是压缩 history 重试，而非立即 abort
                status_code = getattr(api_error, "status_code", None)

                # ── Anthropic Sonnet long-context tier gate ───────────
                # Anthropic returns HTTP 429 "Extra usage is required for
                # long context requests" when a Claude Max (or similar)
                # subscription doesn't include the 1M-context tier.  This
                # is NOT a transient rate limit — retrying or switching
                # credentials won't help.  Reduce context to 200k (the
                # standard tier) and compress.
                # ── Anthropic Sonnet 长上下文 tier 门控 ──
                # Claude Max 等订阅不含 1M 上下文 tier 时 Anthropic 返回 HTTP 429 "Extra usage required"
                # 非 transient 限速 — 重试/换凭证无效；将 context 降到 200k（标准 tier）并压缩
                if classified.reason == FailoverReason.long_context_tier:
                    _reduced_ctx = 200000
                    compressor = self.context_compressor
                    old_ctx = compressor.context_length
                    if old_ctx > _reduced_ctx:
                        compressor.update_model(
                            model=self.model,
                            context_length=_reduced_ctx,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                            provider=self.provider,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).
                        # Context probing 标志 — 仅内置 compressor 设置（插件 engine 自管）
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # Don't persist — this is a subscription-tier
                            # limitation, not a model capability.  If the
                            # user later enables extra usage the 1M limit
                            # should come back automatically.
                            # 不持久化 — 这是订阅 tier 限制，非模型能力；用户日后启用 extra usage 应自动恢复 1M
                            compressor._context_probe_persistable = False
                        self._vprint(
                            f"{self.log_prefix}⚠️  Anthropic long-context tier "
                            f"requires extra usage — reducing context: "
                            f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                            force=True,
                        )

                    compression_attempts += 1
                    if compression_attempts <= max_compression_attempts:
                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        # 压缩创建新 session — 清空 history，使 _flush_messages_to_session_db 写入压缩消息到新 session
                        conversation_history = None
                        if len(messages) < original_len or old_ctx > _reduced_ctx:
                            self._emit_status(
                                f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                f"(was {old_ctx:,}), retrying..."
                            )
                            time.sleep(2)
                            restart_with_compressed_messages = True
                            break
                    # Fall through to normal error handling if compression
                    # is exhausted or didn't help.
                    # 压缩耗尽或无效时 fall through 到正常 error 处理

                # Eager fallback for rate-limit errors (429 or quota exhaustion).
                # When a fallback model is configured, switch immediately instead
                # of burning through retries with exponential backoff -- the
                # primary provider won't recover within the retry window.
                # 429/配额耗尽等限速 error 的积极 fallback
                # 配置了 fallback 时立即切换，而非 backoff 烧尽重试 — 主 provider 不会在重试窗口内恢复
                is_rate_limited = classified.reason in (
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                )
                if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                    # Don't eagerly fallback if credential pool rotation may
                    # still recover.  The pool's retry-then-rotate cycle needs
                    # at least one more attempt to fire — jumping to a fallback
                    # provider here short-circuits it.
                    # 若 credential pool 轮换仍可能恢复，不急于 fallback
                    # pool 的 retry-then-rotate 至少需要再一次机会 — 此处跳 fallback 会短路它
                    pool = self._credential_pool
                    pool_may_recover = pool is not None and pool.has_available()
                    if not pool_may_recover:
                        self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                # ── Nous Portal: record rate limit & skip retries ─────
                # When Nous returns a 429, record the reset time to a
                # shared file so ALL sessions (cron, gateway, auxiliary)
                # know not to pile on.  Then skip further retries —
                # each one burns another RPH request and deepens the
                # rate limit hole.  The retry loop's top-of-iteration
                # guard will catch this on the next pass and try
                # fallback or bail with a clear message.
                # ── Nous Portal：记录限速并跳过重试 ──
                # Nous 返回 429 时将 reset 时间写入共享文件，所有 session（cron、gateway、auxiliary）知晓勿叠加
                # 然后跳过后续重试 — 每次再烧一个 RPH；重试循环顶部 guard 下次会 fallback 或清晰 bail
                if (
                    is_rate_limited
                    and self.provider == "nous"
                    and classified.reason == FailoverReason.rate_limit
                    and not recovered_with_pool
                ):
                    try:
                        from agent.nous_rate_guard import record_nous_rate_limit
                        _err_resp = getattr(api_error, "response", None)
                        _err_hdrs = (
                            getattr(_err_resp, "headers", None)
                            if _err_resp else None
                        )
                        record_nous_rate_limit(
                            headers=_err_hdrs,
                            error_context=error_context,
                        )
                    except Exception:
                        pass
                    # Skip straight to max_retries — the top-of-loop
                    # guard will handle fallback or bail cleanly.
                    # 直接跳到 max_retries — 循环顶部 guard 会处理 fallback 或干净退出
                    retry_count = max_retries
                    continue

                is_payload_too_large = (
                    classified.reason == FailoverReason.payload_too_large
                )

                if is_payload_too_large:
                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                        self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                        self._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                    original_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history
                    # so _flush_messages_to_session_db writes compressed
                    # messages to the new session, not skipping them.
                    # 压缩创建新 session — 清空 history，使 _flush_messages_to_session_db 写入压缩消息
                    conversation_history = None

                    if len(messages) < original_len:
                        self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # Brief pause between compression retries | 压缩重试间短暂暂停
                        restart_with_compressed_messages = True
                        break
                    else:
                        self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                        self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                        self._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": "Request payload too large (413). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # Check for context-length errors BEFORE generic 4xx handler.
                # The classifier detects context overflow from: explicit error
                # messages, generic 400 + large session heuristic (#1630), and
                # server disconnect + large session pattern (#2153).
                # 在通用 4xx 前先检查 context-length error
                # classifier 从：明确 error 消息、generic 400 + 大 session 启发式 (#1630)、
                # server disconnect + 大 session 模式 (#2153) 检测 overflow
                is_context_length_error = (
                    classified.reason == FailoverReason.context_overflow
                )

                if is_context_length_error:
                    compressor = self.context_compressor
                    old_ctx = compressor.context_length

                    # ── Distinguish two very different errors ───────────
                    # 1. "Prompt too long": the INPUT exceeds the context window.
                    #    Fix: reduce context_length + compress history.
                    # 2. "max_tokens too large": input is fine, but
                    #    input_tokens + requested max_tokens > context_window.
                    #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
                    #    Do NOT shrink context_length — the window is unchanged.
                    #
                    # Note: max_tokens = output token cap (one response).
                    #       context_length = total window (input + output combined).
                    # ── 区分两种截然不同的 error ──
                    # 1. "Prompt too long"：INPUT 超上下文窗口 → 降 context_length + 压缩 history
                    # 2. "max_tokens too large"：input 正常，但 input_tokens + max_tokens > context_window
                    #    → 降 max_tokens（OUTPUT 上限）；勿缩小 context_length — 窗口未变
                    #
                    # max_tokens = 单次响应 output 上限；context_length = input+output 总窗口
                    available_out = parse_available_output_tokens_from_error(error_msg)
                    if available_out is not None:
                        # Error is purely about the output cap being too large.
                        # Cap output to the available space and retry without
                        # touching context_length or triggering compression.
                        # error 纯关于 output cap 过大 — 限制 output 到可用空间重试
                        # 不碰 context_length，不触发压缩
                        safe_out = max(1, available_out - 64)  # small safety margin | 小安全边距
                        self._ephemeral_max_output_tokens = safe_out
                        self._vprint(
                            f"{self.log_prefix}⚠️  Output cap too large for current prompt — "
                            f"retrying with max_tokens={safe_out:,} "
                            f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})",
                            force=True,
                        )
                        # Still count against compression_attempts so we don't
                        # loop forever if the error keeps recurring.
                        # 仍计入 compression_attempts，避免 error 反复出现时无限循环
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        restart_with_compressed_messages = True
                        break

                    # Error is about the INPUT being too large — reduce context_length.
                    # Try to parse the actual limit from the error message
                    # error 关于 INPUT 过大 — 降低 context_length；尝试从 error 消息解析实际限制
                    parsed_limit = parse_context_limit_from_error(error_msg)
                    if parsed_limit and parsed_limit < old_ctx:
                        new_ctx = parsed_limit
                        self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                    else:
                        # Step down to the next probe tier
                        # 降到下一个 probe tier
                        new_ctx = get_next_probe_tier(old_ctx)

                    if new_ctx and new_ctx < old_ctx:
                        compressor.update_model(
                            model=self.model,
                            context_length=new_ctx,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                            provider=self.provider,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).
                        # Context probing 标志 — 仅内置 compressor（插件 engine 自管）
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # Only persist limits parsed from the provider's
                            # error message (a real number).  Guessed fallback
                            # tiers from get_next_probe_tier() should stay
                            # in-memory only — persisting them pollutes the
                            # cache with wrong values.
                            # 仅持久化从 provider error 解析的真实数字；get_next_probe_tier() 猜测的 tier
                            # 仅内存 — 持久化会污染 cache
                            compressor._context_probe_persistable = bool(
                                parsed_limit and parsed_limit == new_ctx
                            )
                        self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                    else:
                        self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                        self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                        self._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                    original_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history
                    # so _flush_messages_to_session_db writes compressed
                    # messages to the new session, not skipping them.
                    # 压缩创建新 session — 清空 history
                    conversation_history = None

                    if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # Brief pause between compression retries | 压缩重试间短暂暂停
                        restart_with_compressed_messages = True
                        break
                    else:
                        # Can't compress further and already at minimum tier
                        # 无法再压缩且已在最低 tier
                        self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                        self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                        logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                        self._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # Check for non-retryable client errors.  The classifier
                # already accounts for 413, 429, 529 (transient), context
                # overflow, and generic-400 heuristics.  Local validation
                # errors (ValueError, TypeError) are programming bugs.
                # 检查不可重试 client error；classifier 已处理 413、429、529、overflow 等
                # 本地 validation error（ValueError、TypeError）是编程 bug
                is_local_validation_error = (
                    isinstance(api_error, (ValueError, TypeError))
                    and not isinstance(api_error, UnicodeEncodeError)
                )
                is_client_error = (
                    is_local_validation_error
                    or (
                        not classified.retryable
                        and not classified.should_compress
                        and classified.reason not in (
                            FailoverReason.rate_limit,
                            FailoverReason.billing,
                            FailoverReason.overloaded,
                            FailoverReason.context_overflow,
                            FailoverReason.payload_too_large,
                            FailoverReason.long_context_tier,
                            FailoverReason.thinking_signature,
                        )
                    )
                ) and not is_context_length_error

                if is_client_error:
                    # Try fallback before aborting — a different provider
                    # may not have the same issue (rate limit, auth, etc.)
                    # abort 前尝试 fallback — 不同 provider 可能没有同样问题（限速、auth 等）
                    self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                    if self._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    if api_kwargs is not None:
                        self._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                    self._emit_status(
                        f"❌ Non-retryable error (HTTP {status_code}): "
                        f"{self._summarize_api_error(api_error)}"
                    )
                    self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    # Actionable guidance for common auth errors
                    # 常见 auth error 的可操作指引
                    if classified.is_auth or classified.reason == FailoverReason.billing:
                        if _provider == "openai-codex" and status_code == 401:
                            self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                            self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                            self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                            self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                            self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                            self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                            if "openrouter" in str(_base).lower():
                                self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                    else:
                        self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                    logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
                    # Skip session persistence when the error is likely
                    # context-overflow related (status 400 + large session).
                    # Persisting the failed user message would make the
                    # session even larger, causing the same failure on the
                    # next attempt. (#1630)
                    # error 可能为 context-overflow（400 + 大 session）时跳过 session 持久化
                    # 持久化失败 user 消息会使 session 更大，下次同样失败 (#1630)
                    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                        self._vprint(
                            f"{self.log_prefix}⚠️  Skipping session persistence "
                            f"for large failed session to prevent growth loop.",
                            force=True,
                        )
                    else:
                        self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": str(api_error),
                    }

                if retry_count >= max_retries:
                    # Before falling back, try rebuilding the primary
                    # client once for transient transport errors (stale
                    # connection pool, TCP reset).  Only attempted once
                    # per API call block.
                    # fallback 前尝试重建主 client 一次（transient transport：stale pool、TCP reset）
                    # 每个 API call 块仅尝试一次
                    if not primary_recovery_attempted and self._try_recover_primary_transport(
                        api_error, retry_count=retry_count, max_retries=max_retries,
                    ):
                        primary_recovery_attempted = True
                        retry_count = 0
                        continue
                    # Try fallback before giving up entirely
                    # 彻底放弃前再试 fallback
                    self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                    if self._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    _final_summary = self._summarize_api_error(api_error)
                    if is_rate_limited:
                        self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                    else:
                        self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                    self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

                    # Detect SSE stream-drop pattern (e.g. "Network
                    # connection lost") and surface actionable guidance.
                    # This typically happens when the model generates a
                    # very large tool call (write_file with huge content)
                    # and the proxy/CDN drops the stream mid-response.
                    # 检测 SSE stream-drop 模式（如 "Network connection lost"）并给出可操作指引
                    # 常见于模型生成超大 tool call（write_file 大内容）时 proxy/CDN 中途断流
                    _is_stream_drop = (
                        not getattr(api_error, "status_code", None)
                        and any(p in error_msg for p in (
                            "connection lost", "connection reset",
                            "connection closed", "network connection",
                            "network error", "terminated",
                        ))
                    )
                    if _is_stream_drop:
                        self._vprint(
                            f"{self.log_prefix}   💡 The provider's stream "
                            f"connection keeps dropping. This often happens "
                            f"when the model tries to write a very large "
                            f"file in a single tool call.",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}      Try asking the model "
                            f"to use execute_code with Python's open() for "
                            f"large files, or to write the file in smaller "
                            f"sections.",
                            force=True,
                        )

                    logging.error(
                        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                        self.log_prefix, max_retries, _final_summary,
                        _provider, _model, len(api_messages), f"{approx_tokens:,}",
                    )
                    if api_kwargs is not None:
                        self._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                    self._persist_session(messages, conversation_history)
                    _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                    if _is_stream_drop:
                        _final_response += (
                            "\n\nThe provider's stream connection keeps "
                            "dropping — this often happens when generating "
                            "very large tool call responses (e.g. write_file "
                            "with long content). Try asking me to use "
                            "execute_code with Python's open() for large "
                            "files, or to write in smaller sections."
                        )
                    return {
                        "final_response": _final_response,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _final_summary,
                    }

                # For rate limits, respect the Retry-After header if present
                # 限速时尊重 Retry-After 头（若存在）
                _retry_after = None
                if is_rate_limited:
                    _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                    if _resp_headers and hasattr(_resp_headers, "get"):
                        _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                        if _ra_raw:
                            try:
                                _retry_after = min(int(_ra_raw), 120)  # Cap at 2 minutes | 上限 2 分钟
                            except (TypeError, ValueError):
                                pass
                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                if is_rate_limited:
                    self._emit_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                else:
                    self._emit_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                logger.warning(
                    "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                    wait_time,
                    retry_count,
                    max_retries,
                    self._client_log_context(),
                    api_error,
                )
                # Sleep in small increments so we can respond to interrupts quickly
                # instead of blocking the entire wait_time in one sleep() call
                # 小步 sleep 以便快速响应 interrupt，而非一次 sleep 整个 wait_time
                sleep_end = time.time() + wait_time
                _backoff_touch_counter = 0
                while time.time() < sleep_end:
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    time.sleep(0.2)  # Check interrupt every 200ms | 每 200ms 检查 interrupt
                    # Touch activity every ~30s so the gateway's inactivity
                    # monitor knows we're alive during backoff waits.
                    # 约每 30s touch activity，gateway 不活动监控知道 backoff 期间仍存活
                    _backoff_touch_counter += 1
                    if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s | 约 30 秒
                        self._touch_activity(
                            f"error retry backoff ({retry_count}/{max_retries}), "
                            f"{int(sleep_end - time.time())}s remaining"
                        )
        
        # If the API call was interrupted, skip response processing
        # API 调用被 interrupt 则跳过响应处理
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
            break

        if restart_with_compressed_messages:
            api_call_count -= 1
            self.iteration_budget.refund()
            # Count compression restarts toward the retry limit to prevent
            # infinite loops when compression reduces messages but not enough
            # to fit the context window.
            # 压缩重启计入 retry 限制，防止压缩减消息但仍不够 fit 上下文窗口时的无限循环
            retry_count += 1
            restart_with_compressed_messages = False
            continue

        if restart_with_length_continuation:
            # Progressively boost the output token budget on each retry.
            # Retry 1 → 2× base, retry 2 → 3× base, capped at 32 768.
            # Applies to all providers via _ephemeral_max_output_tokens.
            # 每次重试逐步提升 output token 预算
            # 重试 1 → 2× base，重试 2 → 3× base，上限 32768；通过 _ephemeral_max_output_tokens 适用所有 provider
            _boost_base = self.max_tokens if self.max_tokens else 4096
            _boost = _boost_base * (length_continue_retries + 1)
            self._ephemeral_max_output_tokens = min(_boost, 32768)
            continue

        # Guard: if all retries exhausted without a successful response
        # (e.g. repeated context-length errors that exhausted retry_count),
        # the `response` variable is still None. Break out cleanly.
        # 守卫：所有重试耗尽仍无成功响应（如 context-length error 耗尽 retry_count）
        # `response` 仍为 None 时干净 break
        if response is None:
            _turn_exit_reason = "all_retries_exhausted_no_response"
            print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
            self._persist_session(messages, conversation_history)
            break

        try:
            if self.api_mode == "codex_responses":
                assistant_message, finish_reason = self._normalize_codex_response(response)
            elif self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import normalize_anthropic_response
                assistant_message, finish_reason = normalize_anthropic_response(
                    response, strip_tool_prefix=self._is_anthropic_oauth
                )
            else:
                assistant_message = response.choices[0].message
            
            # Normalize content to string — some OpenAI-compatible servers
            # (llama-server, etc.) return content as a dict or list instead
            # of a plain string, which crashes downstream .strip() calls.
            # 将 content 规范化为 string — 部分 OpenAI 兼容 server（llama-server 等）
            # 返回 dict/list 而非 string，会导致下游 .strip() 崩溃
            if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                raw = assistant_message.content
                if isinstance(raw, dict):
                    assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                elif isinstance(raw, list):
                    # Multimodal content list — extract text parts
                    # 多模态 content list — 提取 text 部分
                    parts = []
                    for part in raw:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                    assistant_message.content = "\n".join(parts)
                else:
                    assistant_message.content = str(raw)

            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                _assistant_text = assistant_message.content or ""
                _invoke_hook(
                    "post_api_request",
                    task_id=effective_task_id,
                    session_id=self.session_id or "",
                    platform=self.platform or "",
                    model=self.model,
                    provider=self.provider,
                    base_url=self.base_url,
                    api_mode=self.api_mode,
                    api_call_count=api_call_count,
                    api_duration=api_duration,
                    finish_reason=finish_reason,
                    message_count=len(api_messages),
                    response_model=getattr(response, "model", None),
                    usage=self._usage_summary_for_api_request_hook(response),
                    assistant_content_chars=len(_assistant_text),
                    assistant_tool_call_count=len(_assistant_tool_calls),
                )
            except Exception:
                pass

            # Handle assistant response
            # 处理 assistant 响应
            if assistant_message.content and not self.quiet_mode:
                if self.verbose_logging:
                    self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                else:
                    self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # Notify progress callback of model's thinking (used by subagent
            # delegation to relay the child's reasoning to the parent display).
            # 通知 progress callback 模型思考（subagent 委托时向父 display  relay 子 agent reasoning）
            if (assistant_message.content and self.tool_progress_callback):
                _think_text = assistant_message.content.strip()
                # Strip reasoning XML tags that shouldn't leak to parent display
                # 剥离不应泄漏到父 display 的 reasoning XML 标签
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                ).strip()
                # For subagents: relay first line to parent display (existing behaviour).
                # For all agents with a structured callback: emit reasoning.available event.
                # subagent：relay 首行到父 display（现有行为）
                # 所有有 structured callback 的 agent：emit reasoning.available 事件
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                if first_line and getattr(self, '_delegate_depth', 0) > 0:
                    try:
                        self.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass
                elif _think_text:
                    try:
                        self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                    except Exception:
                        pass
            
            # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
            # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
            # 检查未闭合的 <REASONING_SCRATCHPAD>（已开未关）
            # 表示模型 reasoning 中途耗尽 output token — 最多重试 2 次
            if has_incomplete_scratchpad(assistant_message.content or ""):
                self._incomplete_scratchpad_retries += 1
                
                self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                
                if self._incomplete_scratchpad_retries <= 2:
                    self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                    # Don't add the broken message, just retry
                    # 不追加 broken message，直接重试
                    continue
                else:
                    # Max retries - discard this turn and save as partial
                    # 达最大重试 — 丢弃本 turn，保存为 partial
                    self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                    self._incomplete_scratchpad_retries = 0
                    
                    rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                    self._cleanup_task_resources(effective_task_id)
                    self._persist_session(messages, conversation_history)
                    
                    return {
                        "final_response": None,
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                    }
            
            # Reset incomplete scratchpad counter on clean response
            # 干净响应时重置 incomplete scratchpad 计数
            self._incomplete_scratchpad_retries = 0

            if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                self._codex_incomplete_retries += 1

                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                interim_has_content = bool((interim_msg.get("content") or "").strip())
                interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

                if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                    last_msg = messages[-1] if messages else None
                    # Duplicate detection: two consecutive incomplete assistant
                    # messages with identical content AND reasoning are collapsed.
                    # For reasoning-only messages (codex_reasoning_items differ but
                    # visible content/reasoning are both empty), we also compare
                    # the encrypted items to avoid silently dropping new state.
                    # 重复检测：两条连续 incomplete assistant 且 content+reasoning 相同则折叠
                    # reasoning-only 消息（codex_reasoning_items 不同但可见 content/reasoning 皆空）
                    # 也比较 encrypted items，避免静默丢弃新 state
                    last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                    interim_codex_items = interim_msg.get("codex_reasoning_items")
                    duplicate_interim = (
                        isinstance(last_msg, dict)
                        and last_msg.get("role") == "assistant"
                        and last_msg.get("finish_reason") == "incomplete"
                        and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                        and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                        and last_codex_items == interim_codex_items
                    )
                    if not duplicate_interim:
                        messages.append(interim_msg)
                        self._emit_interim_assistant_message(interim_msg)

                if self._codex_incomplete_retries < 3:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                    self._session_messages = messages
                    self._save_session_log(messages)
                    continue

                self._codex_incomplete_retries = 0
                self._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Codex response remained incomplete after 3 continuation attempts",
                }
            elif hasattr(self, "_codex_incomplete_retries"):
                self._codex_incomplete_retries = 0
            
            # 判断模型是否要调用工具:
            # 如果要调用工具
            if assistant_message.tool_calls:
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                
                if self.verbose_logging:
                    for tc in assistant_message.tool_calls:
                        logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                
                # Validate tool call names - detect model hallucinations
                # Repair mismatched tool names before validating
                # 校验 tool call 名称 — 检测模型幻觉
                # 校验前先修复不匹配的 tool 名
                for tc in assistant_message.tool_calls:
                    if tc.function.name not in self.valid_tool_names:
                        repaired = self._repair_tool_call(tc.function.name)
                        if repaired:
                            print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                            tc.function.name = repaired
                invalid_tool_calls = [
                    tc.function.name for tc in assistant_message.tool_calls
                    if tc.function.name not in self.valid_tool_names
                ]
                if invalid_tool_calls:
                    # Track retries for invalid tool calls
                    # 跟踪 invalid tool call 重试
                    self._invalid_tool_retries += 1

                    # Return helpful error to model — model can self-correct next turn
                    # 向模型返回 helpful error — 下轮可自纠正
                    available = ", ".join(sorted(self.valid_tool_names))
                    invalid_name = invalid_tool_calls[0]
                    invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                    self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                    if self._invalid_tool_retries >= 3:
                        self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                        self._invalid_tool_retries = 0
                        self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": f"Model generated invalid tool call: {invalid_preview}"
                        }

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    messages.append(assistant_msg)
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                        else:
                            content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": content,
                        })
                    continue
                # Reset retry counter on successful tool call validation
                # tool call 校验成功时重置重试计数
                self._invalid_tool_retries = 0
                
                # Validate tool call arguments are valid JSON
                # Handle empty strings as empty objects (common model quirk)
                # 校验 tool call 参数为合法 JSON
                # 空字符串视为空对象（常见模型 quirk）
                invalid_json_args = []
                for tc in assistant_message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, (dict, list)):
                        tc.function.arguments = json.dumps(args)
                        continue
                    if args is not None and not isinstance(args, str):
                        tc.function.arguments = str(args)
                        args = tc.function.arguments
                    # Treat empty/whitespace strings as empty object
                    # 空/空白字符串视为空对象
                    if not args or not args.strip():
                        tc.function.arguments = "{}"
                        continue
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        invalid_json_args.append((tc.function.name, str(e)))
                
                if invalid_json_args:
                    # Check if the invalid JSON is due to truncation rather
                    # than a model formatting mistake.  Routers sometimes
                    # rewrite finish_reason from "length" to "tool_calls",
                    # hiding the truncation from the length handler above.
                    # Detect truncation: args that don't end with } or ]
                    # (after stripping whitespace) are cut off mid-stream.
                    # 检查 invalid JSON 是否因截断而非格式错误；router 有时把 finish_reason 从 length 改成 tool_calls
                    # 隐藏上方 length handler 的截断；检测：strip 后不以 } 或 ] 结尾的参数为 mid-stream 截断
                    _truncated = any(
                        not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                        for tc in assistant_message.tool_calls
                        if tc.function.name in {n for n, _ in invalid_json_args}
                    )
                    if _truncated:
                        self._vprint(
                            f"{self.log_prefix}⚠️  Truncated tool call arguments detected "
                            f"(finish_reason={finish_reason!r}) — refusing to execute.",
                            force=True,
                        )
                        self._invalid_json_retries = 0
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit",
                        }

                    # Track retries for invalid JSON arguments
                    # 跟踪 invalid JSON 参数重试
                    self._invalid_json_retries += 1

                    tool_name, error_msg = invalid_json_args[0]
                    self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                    if self._invalid_json_retries < 3:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                        # Don't add anything to messages, just retry the API call
                        # 不向 messages 追加任何内容，直接重试 API 调用
                        continue
                    else:
                        # Instead of returning partial, inject tool error results so the model can recover.
                        # Using tool results (not user messages) preserves role alternation.
                        # 不返回 partial，注入 tool error result 让模型恢复
                        # 用 tool result（非 user 消息）保持 role 交替
                        self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                        self._invalid_json_retries = 0  # Reset for next attempt
                        
                        # Append the assistant message with its (broken) tool_calls
                        # 追加带（broken）tool_calls 的 assistant 消息
                        recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(recovery_assistant)
                        
                        # Respond with tool error results for each tool call
                        # 为每个 tool call 响应 tool error result
                        invalid_names = {name for name, _ in invalid_json_args}


                        # 根据输出的调用工具的列表清单，生成工具调用信息
                        for tc in assistant_message.tool_calls:
                            if tc.function.name in invalid_names:
                                err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                tool_result = (
                                    f"Error: Invalid JSON arguments. {err}. "
                                    f"For tools with no required parameters, use an empty object: {{}}. "
                                    f"Please retry with valid JSON."
                                )
                            else:
                                tool_result = "Skipped: other tool call in this response had invalid JSON."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            })
                        continue
                
                # Reset retry counter on successful JSON validation
                # JSON 校验成功时重置重试计数
                self._invalid_json_retries = 0

                # ── Post-call guardrails ──────────────────────────
                # ── 调用后 guardrails ──
                assistant_message.tool_calls = self._cap_delegate_task_calls(
                    assistant_message.tool_calls
                )
                assistant_message.tool_calls = self._deduplicate_tool_calls(
                    assistant_message.tool_calls
                )

                assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                
                # If this turn has both content AND tool_calls, capture the content
                # as a fallback final response. Common pattern: model delivers its
                # answer and calls memory/skill tools as a side-effect in the same
                # turn. If the follow-up turn after tools is empty, we use this.
                # 若本 turn 同时有 content 和 tool_calls，捕获 content 作为 fallback final response
                # 常见模式：模型先给答案再 side-effect 调 memory/skill；若 tool 后 follow-up 为空则用此
                turn_content = assistant_message.content or ""
                if turn_content and self._has_content_after_think_block(turn_content):
                    self._last_content_with_tools = turn_content
                    # Only mute subsequent output when EVERY tool call in
                    # this turn is post-response housekeeping (memory, todo,
                    # skill_manage, etc.).  If any substantive tool is present
                    # (search_files, read_file, write_file, terminal, ...),
                    # keep output visible so the user sees progress.
                    # 仅当本 turn 每个 tool call 都是 post-response housekeeping（memory、todo、skill_manage 等）
                    # 才 mute 后续输出；若有实质 tool（search_files、read_file、write_file、terminal…）保持可见
                    _HOUSEKEEPING_TOOLS = frozenset({
                        "memory", "todo", "skill_manage", "session_search",
                    })
                    _all_housekeeping = all(
                        tc.function.name in _HOUSEKEEPING_TOOLS
                        for tc in assistant_message.tool_calls
                    )
                    self._last_content_tools_all_housekeeping = _all_housekeeping
                    if _all_housekeeping and self._has_stream_consumers():
                        self._mute_post_response = True
                    elif self._should_emit_quiet_tool_messages():
                        clean = self._strip_think_blocks(turn_content).strip()
                        if clean:
                            self._vprint(f"  ┊ 💬 {clean}")
                
                # Pop thinking-only prefill message(s) before appending
                # (tool-call path — same rationale as the final-response path).
                # 追加前 pop thinking-only prefill 消息（tool-call 路径 — 与 final-response 路径同理）
                _had_prefill = False
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()
                    _had_prefill = True

                # Reset prefill counter when tool calls follow a prefill
                # recovery.  Without this, the counter accumulates across
                # the whole conversation — a model that intermittently
                # empties (empty → prefill → tools → empty → prefill →
                # tools) burns both prefill attempts and the third empty
                # gets zero recovery.  Resetting here treats each tool-
                # call success as a fresh start.
                # tool call 跟随 prefill 恢复时重置 prefill 计数
                # 否则计数跨整段对话累积 — empty→prefill→tools→empty… 会烧尽 prefill 且第三次 empty 无恢复
                # 此处重置：每次 tool-call 成功视为新起点
                if _had_prefill:
                    self._thinking_prefill_retries = 0
                    self._empty_content_retries = 0
                # Successful tool execution — reset the post-tool nudge
                # flag so it can fire again if the model goes empty on
                # a LATER tool round.
                # tool 执行成功 — 重置 post-tool nudge 标志，以便后续 tool 轮再次 empty 时可触发
                self._post_tool_empty_retried = False

                messages.append(assistant_msg)
                self._emit_interim_assistant_message(assistant_msg)

                # Close any open streaming display (response box, reasoning
                # box) before tool execution begins.  Intermediate turns may
                # have streamed early content that opened the response box;
                # flushing here prevents it from wrapping tool feed lines.
                # Only signal the display callback — TTS (_stream_callback)
                # should NOT receive None (it uses None as end-of-stream).
                # tool 执行开始前关闭打开的 streaming display（response box、reasoning box）
                # 中间 turn 可能已 stream 早期内容打开 response box；此处 flush 防止包裹 tool feed 行
                # 仅 signal display callback — TTS (_stream_callback) 不应收到 None（None 表示 end-of-stream）
                if self.stream_delta_callback:
                    try:
                        self.stream_delta_callback(None)
                    except Exception:
                        pass

                # 核心：调用工具
                self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                # Reset per-turn retry counters after successful tool
                # execution so a single truncation doesn't poison the
                # entire conversation.
                # tool 执行成功后重置 per-turn 重试计数，避免单次截断污染整段对话
                truncated_tool_call_retries = 0

                # Signal that a paragraph break is needed before the next
                # streamed text.  We don't emit it immediately because
                # multiple consecutive tool iterations would stack up
                # redundant blank lines.  Instead, _fire_stream_delta()
                # will prepend a single "\n\n" the next time real text
                # arrives.
                # 标记下次 stream 文本前需要段落分隔
                # 不立即 emit — 连续多轮 tool 会堆叠多余空行；_fire_stream_delta() 下次有真实文本时 prepend 单个 "\n\n"
                self._stream_needs_break = True

                # Refund the iteration if the ONLY tool(s) called were
                # execute_code (programmatic tool calling).  These are
                # cheap RPC-style calls that shouldn't eat the budget.
                # 若唯一调用的 tool 是 execute_code（程序化 tool calling），退还 iteration
                # 这些是廉价 RPC 式调用，不应消耗预算
                _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                if _tc_names == {"execute_code"}:
                    self.iteration_budget.refund()
                
                # Use real token counts from the API response to decide
                # compression.  prompt_tokens + completion_tokens is the
                # actual context size the provider reported plus the
                # assistant turn — a tight lower bound for the next prompt.
                # Tool results appended above aren't counted yet, but the
                # threshold (default 50%) leaves ample headroom; if tool
                # results push past it, the next API call will report the
                # real total and trigger compression then.
                #
                # If last_prompt_tokens is 0 (stale after API disconnect
                # or provider returned no usage data), fall back to rough
                # estimate to avoid missing compression.  Without this,
                # a session can grow unbounded after disconnects because
                # should_compress(0) never fires.  (#2153)
                # 用 API 响应的真实 token 计数决定是否压缩
                # prompt_tokens + completion_tokens 是 provider 报告的实际上下文 + assistant turn
                # 上方追加的 tool result 尚未计入，但默认 50% 阈值留足 headroom；超出则下次 API 报告真实总量触发压缩
                #
                # last_prompt_tokens 为 0（API 断开后 stale 或 provider 无 usage）时回退 rough 估算
                # 否则 should_compress(0) 永不触发，断开后 session 可能无限增长 (#2153)
                _compressor = self.context_compressor
                if _compressor.last_prompt_tokens > 0:
                    # Only use prompt_tokens — completion/reasoning
                    # tokens don't consume context window space.
                    # Thinking models (GLM-5.1, QwQ, DeepSeek R1)
                    # inflate completion_tokens with reasoning,
                    # causing premature compression.  (#12026)
                    # 仅用 prompt_tokens — completion/reasoning token 不占上下文窗口
                    # 思考模型（GLM-5.1、QwQ、DeepSeek R1）completion_tokens 含 reasoning 会导致过早压缩 (#12026)
                    _real_tokens = _compressor.last_prompt_tokens
                else:
                    _real_tokens = estimate_messages_tokens_rough(messages)

                # 压缩上下文
                if self.compression_enabled and _compressor.should_compress(_real_tokens):
                    self._safe_print("  ⟳ compacting context…")
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message,
                        approx_tokens=self.context_compressor.last_prompt_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history so
                    # _flush_messages_to_session_db writes compressed messages
                    # to the new session (see preflight compression comment).
                    # 压缩创建新 session — 清空 history（见 preflight compression 注释）
                    conversation_history = None
                
                # Save session log incrementally (so progress is visible even if interrupted)
                # 增量保存 session log（中断时仍可见进度）
                self._session_messages = messages
                self._save_session_log(messages)
                
                # Continue loop for next response
                # 继续循环等待下一轮响应
                continue
            
            else:  # 模型没有调用工具
                # No tool calls - this is the final response
                # 无 tool call — 这是最终响应
                final_response = assistant_message.content or ""
                
                # Fix: unmute output when entering the no-tool-call branch
                # so the user can see empty-response warnings and recovery
                # status messages.  _mute_post_response was set during a
                # prior housekeeping tool turn and should not silence the
                # final response path.
                # 修复：进入无-tool-call 分支时 unmute 输出
                # 使用户能看到 empty-response 警告和恢复状态；_mute_post_response 不应 silence 最终响应路径
                self._mute_post_response = False
                
                # Check if response only has think block with no actual content after it
                # 检查响应是否仅有 think block 而无之后实际 content
                if not self._has_content_after_think_block(final_response):
                    # ── Partial stream recovery ─────────────────────
                    # If content was already streamed to the user before
                    # the connection died, use it as the final response
                    # instead of falling through to prior-turn fallback
                    # or wasting API calls on retries.
                    # ── 部分 stream 恢复 ──
                    # 连接断开前已向用户 stream 的内容，用作 final response
                    # 而非 fall through 到 prior-turn fallback 或浪费 API 重试
                    _partial_streamed = (
                        getattr(self, "_current_streamed_assistant_text", "") or ""
                    )
                    if self._has_content_after_think_block(_partial_streamed):
                        _turn_exit_reason = "partial_stream_recovery"
                        _recovered = self._strip_think_blocks(_partial_streamed).strip()
                        logger.info(
                            "Partial stream content delivered (%d chars) "
                            "— using as final response",
                            len(_recovered),
                        )
                        self._emit_status(
                            "↻ Stream interrupted — using delivered content "
                            "as final response"
                        )
                        final_response = _recovered
                        self._response_was_previewed = True
                        break

                    # If the previous turn already delivered real content alongside
                    # HOUSEKEEPING tool calls (e.g. "You're welcome!" + memory save),
                    # the model has nothing more to say. Use the earlier content
                    # immediately instead of wasting API calls on retries.
                    # NOTE: Only use this shortcut when ALL tools in that turn were
                    # housekeeping (memory, todo, etc.).  When substantive tools
                    # were called (terminal, search_files, etc.), the content was
                    # likely mid-task narration ("I'll scan the directory...") and
                    # the empty follow-up means the model choked — let the
                    # post-tool nudge below handle that instead of exiting early.
                    # 若上一 turn 已在 HOUSEKEEPING tool  alongside 交付真实 content（如 "You're welcome!" + memory save）
                    # 模型无需再说 — 立即用 earlier content，避免浪费 API 重试
                    # 注意：仅当该 turn 所有 tool 都是 housekeeping；实质 tool 时 content 可能是 mid-task 旁白
                    # empty follow-up 表示模型 choked — 让下方 post-tool nudge 处理，勿 early exit
                    fallback = getattr(self, '_last_content_with_tools', None)
                    if fallback and getattr(self, '_last_content_tools_all_housekeeping', False):
                        _turn_exit_reason = "fallback_prior_turn_content"
                        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                        self._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                        self._last_content_with_tools = None
                        self._last_content_tools_all_housekeeping = False
                        self._empty_content_retries = 0
                        # Do NOT modify the assistant message content — the
                        # old code injected "Calling the X tools..." which
                        # poisoned the conversation history.  Just use the
                        # fallback text as the final response and break.
                        # 勿修改 assistant message content — 旧代码注入 "Calling the X tools..." 会污染 history
                        # 仅用 fallback 文本作 final response 并 break
                        final_response = self._strip_think_blocks(fallback).strip()
                        self._response_was_previewed = True
                        break

                    # ── Post-tool-call empty response nudge ───────────
                    # The model returned empty after executing tool calls.
                    # This covers two cases:
                    #  (a) No prior-turn content at all — model went silent
                    #  (b) Prior turn had content + SUBSTANTIVE tools (the
                    #      fallback above was skipped because the content
                    #      was mid-task narration, not a final answer)
                    # Instead of giving up, nudge the model to continue by
                    # appending a user-level hint.  This is the #9400 case:
                    # weaker models (mimo-v2-pro, GLM-5, etc.) sometimes
                    # return empty after tool results instead of continuing
                    # to the next step.  One retry with a nudge usually
                    # fixes it.
                    # ── Tool 调用后 empty response nudge ──
                    # 模型执行 tool 后返回 empty；覆盖：(a) 无 prior content — 模型沉默
                    # (b) prior 有 content + 实质 tool（上方 fallback 跳过）— content 是 mid-task 旁白非最终答案
                    # 不放弃，追加 user 级 hint nudge；#9400：弱模型（mimo-v2-pro、GLM-5）tool result 后 empty
                    # 一次 nudge 重试通常可修复
                    _prior_was_tool = any(
                        m.get("role") == "tool"
                        for m in messages[-5:]  # check recent messages | 检查最近消息
                    )
                    if (
                        _prior_was_tool
                        and not getattr(self, "_post_tool_empty_retried", False)
                    ):
                        self._post_tool_empty_retried = True
                        # Clear stale narration so it doesn't resurface
                        # on a later empty response after the nudge.
                        # 清除 stale 旁白，避免 nudge 后 later empty 再次 surfaced
                        self._last_content_with_tools = None
                        self._last_content_tools_all_housekeeping = False
                        logger.info(
                            "Empty response after tool calls — nudging model "
                            "to continue processing"
                        )
                        self._emit_status(
                            "⚠️ Model returned empty after tool calls — "
                            "nudging to continue"
                        )
                        # Append the empty assistant message first so the
                        # message sequence stays valid:
                        #   tool(result) → assistant("(empty)") → user(nudge)
                        # Without this, we'd have tool → user which most
                        # APIs reject as an invalid sequence.
                        # 先追加 empty assistant 消息以保持合法序列：
                        #   tool(result) → assistant("(empty)") → user(nudge)
                        # 否则 tool → user 大多数 API 拒绝
                        _nudge_msg = self._build_assistant_message(assistant_message, finish_reason)
                        _nudge_msg["content"] = "(empty)"
                        messages.append(_nudge_msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                "You just executed tool calls but returned an "
                                "empty response. Please process the tool "
                                "results above and continue with the task."
                            ),
                        })
                        continue

                    # ── Thinking-only prefill continuation ──────────
                    # The model produced structured reasoning (via API
                    # fields) but no visible text content.  Rather than
                    # giving up, append the assistant message as-is and
                    # continue — the model will see its own reasoning
                    # on the next turn and produce the text portion.
                    # Inspired by clawdbot's "incomplete-text" recovery.
                    # ── 仅 thinking 的 prefill 续写 ──
                    # 模型通过 API 字段产生 structured reasoning 但无可见文本
                    # 不放弃，原样 append assistant 消息并 continue — 下轮模型看到自己的 reasoning 再产出文本
                    # 灵感来自 clawdbot "incomplete-text" 恢复
                    _has_structured = bool(
                        getattr(assistant_message, "reasoning", None)
                        or getattr(assistant_message, "reasoning_content", None)
                        or getattr(assistant_message, "reasoning_details", None)
                    )
                    if _has_structured and self._thinking_prefill_retries < 2:
                        self._thinking_prefill_retries += 1
                        logger.info(
                            "Thinking-only response (no visible content) — "
                            "prefilling to continue (%d/2)",
                            self._thinking_prefill_retries,
                        )
                        self._emit_status(
                            f"↻ Thinking-only response — prefilling to continue "
                            f"({self._thinking_prefill_retries}/2)"
                        )
                        interim_msg = self._build_assistant_message(
                            assistant_message, "incomplete"
                        )
                        interim_msg["_thinking_prefill"] = True
                        messages.append(interim_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    # ── Empty response retry ──────────────────────
                    # Model returned nothing usable.  Retry up to 3
                    # times before attempting fallback.  This covers
                    # both truly empty responses (no content, no
                    # reasoning) AND reasoning-only responses after
                    # prefill exhaustion — models like mimo-v2-pro
                    # always populate reasoning fields via OpenRouter,
                    # so the old `not _has_structured` guard blocked
                    # retries for every reasoning model after prefill.
                    # ── Empty response 重试 ──
                    # 模型返回无可用内容；fallback 前最多重试 3 次
                    # 覆盖真正 empty 和 prefill 耗尽后的 reasoning-only — mimo-v2-pro 等经 OpenRouter 总有 reasoning 字段
                    # 旧 `not _has_structured` guard 会 block 所有 reasoning 模型在 prefill 后的重试
                    _truly_empty = not self._strip_think_blocks(
                        final_response
                    ).strip()
                    _prefill_exhausted = (
                        _has_structured
                        and self._thinking_prefill_retries >= 2
                    )
                    if _truly_empty and (not _has_structured or _prefill_exhausted) and self._empty_content_retries < 3:
                        self._empty_content_retries += 1
                        logger.warning(
                            "Empty response (no content or reasoning) — "
                            "retry %d/3 (model=%s)",
                            self._empty_content_retries, self.model,
                        )
                        self._emit_status(
                            f"⚠️ Empty response from model — retrying "
                            f"({self._empty_content_retries}/3)"
                        )
                        continue

                    # ── Exhausted retries — try fallback provider ──
                    # Before giving up with "(empty)", attempt to
                    # switch to the next provider in the fallback
                    # chain.  This covers the case where a model
                    # (e.g. GLM-4.5-Air) consistently returns empty
                    # due to context degradation or provider issues.
                    # ── 重试耗尽 — 尝试 fallback provider ──
                    # 放弃 "(empty)" 前尝试 fallback chain
                    # 覆盖模型（如 GLM-4.5-Air）因上下文退化/provider 问题持续 empty
                    if _truly_empty and self._fallback_chain:
                        logger.warning(
                            "Empty response after %d retries — "
                            "attempting fallback (model=%s, provider=%s)",
                            self._empty_content_retries, self.model,
                            self.provider,
                        )
                        self._emit_status(
                            "⚠️ Model returning empty responses — "
                            "switching to fallback provider..."
                        )
                        if self._try_activate_fallback():
                            self._empty_content_retries = 0
                            self._emit_status(
                                f"↻ Switched to fallback: {self.model} "
                                f"({self.provider})"
                            )
                            logger.info(
                                "Fallback activated after empty responses: "
                                "now using %s on %s",
                                self.model, self.provider,
                            )
                            continue

                    # Exhausted retries and fallback chain (or no
                    # fallback configured).  Fall through to the
                    # "(empty)" terminal.
                    # 重试和 fallback 均耗尽（或未配置 fallback）— fall through 到 "(empty)" 终态
                    _turn_exit_reason = "empty_response_exhausted"
                    reasoning_text = self._extract_reasoning(assistant_message)
                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    assistant_msg["content"] = "(empty)"
                    messages.append(assistant_msg)

                    if reasoning_text:
                        reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                        logger.warning(
                            "Reasoning-only response (no visible content) "
                            "after exhausting retries and fallback. "
                            "Reasoning: %s", reasoning_preview,
                        )
                        self._emit_status(
                            "⚠️ Model produced reasoning but no visible "
                            "response after all retries. Returning empty."
                        )
                    else:
                        logger.warning(
                            "Empty response (no content or reasoning) "
                            "after %d retries. No fallback available. "
                            "model=%s provider=%s",
                            self._empty_content_retries, self.model,
                            self.provider,
                        )
                        self._emit_status(
                            "❌ Model returned no content after all retries"
                            + (" and fallback attempts." if self._fallback_chain else
                                ". No fallback providers configured.")
                        )

                    final_response = "(empty)"
                    break
                
                # Reset retry counter/signature on successful content
                # 成功获得 content 时重置重试计数/签名
                self._empty_content_retries = 0
                self._thinking_prefill_retries = 0

                if (
                    self.api_mode == "codex_responses"
                    and self.valid_tool_names
                    and codex_ack_continuations < 2
                    and self._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                    )
                ):
                    codex_ack_continuations += 1
                    interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                    messages.append(interim_msg)
                    self._emit_interim_assistant_message(interim_msg)

                    continue_msg = {
                        "role": "user",
                        "content": (
                            "[System: Continue now. Execute the required tool calls and only "
                            "send your final answer after completing the task.]"
                        ),
                    }
                    messages.append(continue_msg)
                    self._session_messages = messages
                    self._save_session_log(messages)
                    continue

                codex_ack_continuations = 0

                if truncated_response_prefix:
                    final_response = truncated_response_prefix + final_response
                    truncated_response_prefix = ""
                    length_continue_retries = 0
                
                # Strip <think> blocks from user-facing response (keep raw in messages for trajectory)
                # 从用户可见 final_response 剥离 <think>（messages 保留 raw 供 trajectory）
                final_response = self._strip_think_blocks(final_response).strip()
                
                final_msg = self._build_assistant_message(assistant_message, finish_reason)

                # Pop thinking-only prefill message(s) before appending
                # the final response.  This avoids consecutive assistant
                # messages which break strict-alternation providers
                # (Anthropic Messages API) and keeps history clean.
                # 追加最终响应前 pop thinking-only prefill 消息
                # 避免连续 assistant 消息破坏严格交替 provider（Anthropic Messages API），保持 history 干净
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()

                messages.append(final_msg)
                
                _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                if not self.quiet_mode:
                    self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                break
            
        except Exception as e:
            error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
            try:
                print(f"❌ {error_msg}")
            except (OSError, ValueError):
                logger.error(error_msg)
            
            logger.debug("Outer loop error in API call #%d", api_call_count, exc_info=True)
            
            # If an assistant message with tool_calls was already appended,
            # the API expects a role="tool" result for every tool_call_id.
            # Fill in error results for any that weren't answered yet.
            # 若已 append 带 tool_calls 的 assistant 消息，API 期望每个 tool_call_id 有 role=tool result
            # 为尚未应答的填充 error result
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if not isinstance(msg, dict):
                    break
                if msg.get("role") == "tool":
                    continue
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    answered_ids = {
                        m["tool_call_id"]
                        for m in messages[idx + 1:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    }
                    for tc in msg["tool_calls"]:
                        if not tc or not isinstance(tc, dict): continue
                        if tc["id"] not in answered_ids:
                            err_msg = {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": f"Error executing tool: {error_msg}",
                            }
                            messages.append(err_msg)
                break
            
            # Non-tool errors don't need a synthetic message injected.
            # The error is already printed to the user (line above), and
            # the retry loop continues.  Injecting a fake user/assistant
            # message pollutes history, burns tokens, and risks violating
            # role-alternation invariants.
            # 非 tool error 无需注入 synthetic 消息
            # error 已打印给用户，重试循环继续；假 user/assistant 会污染 history、烧 token、破坏 role 交替

            # If we're near the limit, break to avoid infinite loops
            # 接近 limit 时 break，避免无限循环
            if api_call_count >= self.max_iterations - 1:
                _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                # Append as assistant so the history stays valid for
                # session resume (avoids consecutive user messages).
                # 追加为 assistant，使 history 可 session resume（避免连续 user 消息）
                messages.append({"role": "assistant", "content": final_response})
                break
    
    # 处理final_response
    if final_response is None and (
        api_call_count >= self.max_iterations
        or self.iteration_budget.remaining <= 0
    ):
        # Budget exhausted — ask the model for a summary via one extra
        # API call with tools stripped.  _handle_max_iterations injects a
        # user message and makes a single toolless request.
        # 预算耗尽 — 通过一次额外无 tool 的 API 调用让模型摘要
        # _handle_max_iterations 注入 user 消息并发起单次 toolless 请求
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{self.max_iterations})"
        self._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
            "— asking model to summarise"
        )
        if not self.quiet_mode:
            self._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                "— requesting summary..."
            )
        final_response = self._handle_max_iterations(messages, api_call_count)
    
    # Determine if conversation completed successfully
    # 判断对话是否成功完成
    completed = final_response is not None and api_call_count < self.max_iterations

    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    # 若启用则保存 trajectory；``user_message`` 可能是多模态 parts 列表，trajectory 需要纯字符串
    self._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)

    # Clean up VM and browser for this task after conversation completes
    # 对话完成后清理本 task 的 VM 和 browser
    self._cleanup_task_resources(effective_task_id)

    
    # =======================阶段5: 回合收尾 (Turn Teardown)===============================
    result = _handle_teardown(messages, conversation_history)

    return result


# =======================阶段5: 回合收尾 (Turn Teardown)===============================

    _handle_teardown()


def _handle_teardown(messages: list[dict[str, Any]], conversation_history: list[dict[str, Any]])->dict[str, Any]:
    
    # Persist session to both JSON log and SQLite
    # 持久化 session 到 JSON log 和 SQLite
    self._persist_session(messages, conversation_history)

    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always logged at INFO so agent.log captures WHY every turn ended.
    # When the last message is a tool result (agent was mid-work), log
    # at WARNING — this is the "just stops" scenario users report.
    # ── 回合退出诊断日志 ──
    # 始终 INFO 级别写入 agent.log，记录每回合结束原因
    # 最后一条是 tool result（agent 工作中）时 WARNING — 用户报告的 "突然停止" 场景
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        # 回溯找到带 tool call 的 assistant 消息
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = self.iteration_budget.used if self.iteration_budget else 0
    _budget_max = self.iteration_budget.max_total if self.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, self.model, api_call_count, self.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        self.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        # Agent 工作中 — "突然停止" 场景
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # Plugin hook: post_llm_call
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can use this to persist conversation data (e.g. sync
    # to an external memory system).
    # 处理最终的输出
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # Extract reasoning from the last assistant message (if any)
    # 从最后一条 assistant 消息提取 reasoning（若有）
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # Build result with interrupt info if applicable
    # 给出最终的答案：final_response
    # 构建含 interrupt 信息的结果
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "partial": False,  # True only when stopped due to invalid tool calls | 仅因 invalid tool call 停止时为 True
        "interrupted": interrupted,
        "response_previewed": getattr(self, "_response_was_previewed", False),
        "model": self.model,
        "provider": self.provider,
        "base_url": self.base_url,
        "input_tokens": self.session_input_tokens,
        "output_tokens": self.session_output_tokens,
        "cache_read_tokens": self.session_cache_read_tokens,
        "cache_write_tokens": self.session_cache_write_tokens,
        "reasoning_tokens": self.session_reasoning_tokens,
        "prompt_tokens": self.session_prompt_tokens,
        "completion_tokens": self.session_completion_tokens,
        "total_tokens": self.session_total_tokens,
        "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": self.session_estimated_cost_usd,
        "cost_status": self.session_cost_status,
        "cost_source": self.session_cost_source,
    }
    # If a /steer landed after the final assistant turn (no more tool
    # batches to drain into), hand it back to the caller so it can be
    # delivered as the next user turn instead of being silently lost.
    # 若 /steer 在最终 assistant turn 之后到达（无更多 tool batch 可 drain），
    # 交还 caller 作为下一 user turn，而非静默丢失
    _leftover_steer = self._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    self._response_was_previewed = False
    
    # Include interrupt message if one triggered the interrupt
    # 若 interrupt 由消息触发，包含 interrupt_message
    if interrupted and self._interrupt_message:
        result["interrupt_message"] = self._interrupt_message
    
    # Clear interrupt state after handling
    # 处理完毕后清除 interrupt 状态
    self.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    # 清除 stream callback，避免泄漏到后续调用
    self._stream_callback = None

    # Check skill trigger NOW — based on how many tool iterations THIS turn used.
    # 此刻检查 skill 触发 — 基于本回合工具迭代次数
    _should_review_skills = False
    if (self._skill_nudge_interval > 0
            and self._iters_since_skill >= self._skill_nudge_interval
            and "skill_manage" in self.valid_tool_names):
        _should_review_skills = True
        self._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    # Use original_user_message (clean input) — user_message may contain
    # injected skill content that bloats / breaks provider queries.
    # 外部 memory provider：同步已完成 turn + 排队下次 prefetch
    # 使用 original_user_message（干净输入）
    if self._memory_manager and final_response and original_user_message:
        try:
            self._memory_manager.sync_all(original_user_message, final_response)
            self._memory_manager.queue_prefetch_all(original_user_message)
        except Exception:
            pass

    # Background memory/skill review — runs AFTER the response is delivered
    # so it never competes with the user's task for model attention.
    # 后台 memory/skill review — 在响应交付后运行
    # 不与用户任务争用模型注意力
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            self._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort | 后台 review 尽力而为

    # Note: Memory provider on_session_end() + shutdown_all() are NOT
    # called here — run_conversation() is called once per user message in
    # multi-turn sessions. Shutting down after every turn would kill the
    # provider before the second message. Actual session-end cleanup is
    # handled by the CLI (atexit / /reset) and gateway (session expiry /
    # _reset_session).
    # 注意：Memory provider on_session_end() + shutdown_all() 此处不调用
    # run_conversation() 在多轮 session 中每条 user 消息调用一次；每 turn shutdown 会在第二条消息前 kill provider
    # 实际 session 结束清理由 CLI（atexit / /reset）和 gateway（session 过期 / _reset_session）处理

    # Plugin hook: on_session_end
    # Fired at the very end of every run_conversation call.
    # Plugins can use this for cleanup, flushing buffers, etc.
    # 插件钩子：on_session_end
    # 每次 run_conversation 调用最末尾触发；插件可做 cleanup、flush buffer 等
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=self.session_id,
            completed=completed,
            interrupted=interrupted,
            model=self.model,
            platform=getattr(self, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    return result

def init_conversation():

    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    # Installed once, transparent when streams are healthy, prevents crash on write.
    # 防止 broken pipe 导致 stdio 写入 OSError 崩溃（systemd/无头/daemon 环境）
    # 只安装一次；流正常时透明，写入失败时不会崩溃

    _install_safe_stdio()

    # Tag all log records on this thread with the session ID so
    # ``hermes logs --session <id>`` can filter a single conversation.
    # 为本线程所有日志记录打上 session ID，便于 `hermes logs --session <id>` 过滤
    # 单条会话的日志
    from hermes_logging import set_session_context
    set_session_context(self.session_id)

    # If the previous turn activated fallback, restore the primary
    # runtime so this turn gets a fresh attempt with the preferred model.
    # No-op when _fallback_activated is False (gateway, first turn, etc.).
    # 若上一回合激活了 fallback，本回合恢复主运行时，重新尝试首选模型
    # 当 _fallback_activated 为 False 时无操作（gateway、首回合等）
    self._restore_primary_runtime()

    # Sanitize surrogate characters from user input.  Clipboard paste from
    # rich-text editors (Google Docs, Word, etc.) can inject lone surrogates
    # that are invalid UTF-8 and crash JSON serialization in the OpenAI SDK.
    # 清洗用户输入中的孤立 surrogate 字符；从富文本编辑器粘贴可能带入无效 UTF-8
    # 导致 OpenAI SDK JSON 序列化崩溃
    if isinstance(user_message, str):
        user_message = _sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = _sanitize_surrogates(persist_user_message)

    # Strip leaked <memory-context> blocks from user input.  When Honcho's
    # saveMessages persists a turn that included injected context, the block
    # can reappear in the next turn's user message via message history.
    # Stripping here prevents stale memory tags from leaking into the
    # conversation and being visible to the user or the model as user text.
    # 剥离用户输入中泄漏的 <memory-context> 块；Honcho saveMessages 可能把注入上下文
    # 带入下一回合 history，此处剥离防止 stale memory 标签泄漏给用户或模型
    if isinstance(user_message, str):
        user_message = sanitize_context(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_context(persist_user_message)

    # Store stream callback for _interruptible_api_call to pick up
    # 保存 stream_callback，供 _interruptible_api_call 使用
    self._stream_callback = stream_callback
    self._persist_user_message_idx = None
    self._persist_user_message_override = persist_user_message
    # Generate unique task_id if not provided to isolate VMs between concurrent tasks
    # 未提供 task_id 时自动生成，隔离并发任务之间的 VM
    effective_task_id = task_id or str(uuid.uuid4())
    
    # Reset retry counters and iteration budget at the start of each turn
    # so subagent usage from a previous turn doesn't eat into the next one.
    # 每回合开始时重置重试计数器和迭代预算
    # 避免上一回合 subagent 消耗影响本回合
    self._invalid_tool_retries = 0
    self._invalid_json_retries = 0
    self._empty_content_retries = 0
    self._incomplete_scratchpad_retries = 0
    self._codex_incomplete_retries = 0
    self._thinking_prefill_retries = 0
    self._post_tool_empty_retried = False
    self._last_content_with_tools = None
    self._last_content_tools_all_housekeeping = False
    self._mute_post_response = False
    self._unicode_sanitization_passes = 0

    # Pre-turn connection health check: detect and clean up dead TCP
    # connections left over from provider outages or dropped streams.
    # This prevents the next API call from hanging on a zombie socket.
    # 回合前连接健康检查：清理 provider 中断/流断开遗留的死 TCP 连接
    # 防止下次 API 调用在僵尸 socket 上挂起
    if self.api_mode != "anthropic_messages":
        try:
            if self._cleanup_dead_connections():
                self._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # Replay compression warning through status_callback for gateway
    # platforms (the callback was not wired during __init__).
    # 通过 status_callback 重播压缩警告（gateway 平台 __init__ 时未接线 callback）
    if self._compression_warning:
        self._replay_compression_warning()
        self._compression_warning = None  # send once | 只发送一次

    # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
    # They are initialized in __init__ and must persist across run_conversation
    # calls so that nudge logic accumulates correctly in CLI mode.
    # 注意：_turns_since_memory 和 _iters_since_skill 此处不重置
    # 它们在 __init__ 初始化，须跨 run_conversation 调用累积，CLI nudge 逻辑才正确
    self.iteration_budget = IterationBudget(self.max_iterations)

    # Log conversation turn start for debugging/observability
    # 记录回合开始，便于调试/可观测性
    _preview_text = _summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        self.session_id or "none", self.model, self.provider or "unknown",
        self.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list)
    # 初始化对话（复制列表，避免修改调用方传入的 history）
    messages = list(conversation_history) if conversation_history else []

    # Hydrate todo store from conversation history (gateway creates a fresh
    # AIAgent per message, so the in-memory store is empty -- we need to
    # recover the todo state from the most recent todo tool response in history)
    # 从 history 恢复 todo store（gateway 每条消息新建 AIAgent，内存 store 为空
    # 需从历史中最近的 todo 工具响应恢复状态）
    if conversation_history and not self._todo_store.has_items():
        self._hydrate_todo_store(conversation_history)

    return None

# 会话状态准备
def prepare_session():
        # Prefill messages (few-shot priming) are injected at API-call time only,
    # never stored in the messages list. This keeps them ephemeral: they won't
    # be saved to session DB, session logs, or batch trajectories, but they're
    # automatically re-applied on every API call (including session continuations).
    # Prefill 消息（few-shot 引导）仅在 API 调用时注入，不写入 messages 列表
    # 保持 ephemeral：不存 session DB/日志/trajectory，但每轮 API 调用都会重新应用
    
    # Track user turns for memory flush and periodic nudge logic
    # 跟踪用户回合数，用于 memory flush 和周期性 nudge
    self._user_turn_count += 1

    # Preserve the original user message (no nudge injection).
    # 保留原始用户消息（不含 nudge 注入）
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # Track memory nudge trigger (turn-based, checked here).
    # Skill trigger is checked AFTER the agent loop completes, based on
    # how many tool iterations THIS turn used.
    # 在此检查 memory nudge 触发（按回合）
    # skill 触发在主循环结束后检查，依据本回合工具迭代次数
    _should_review_memory = False
    if (self._memory_nudge_interval > 0
            and "memory" in self.valid_tool_names
            and self._memory_store):
        self._turns_since_memory += 1
        if self._turns_since_memory >= self._memory_nudge_interval:
            _should_review_memory = True
            self._turns_since_memory = 0

    # Add user message
    # 追加用户消息
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    self._persist_user_message_idx = current_turn_user_idx
    
    if not self.quiet_mode:
        _print_preview = _summarize_user_message_for_log(user_message)
        self._safe_print(f"💬 Starting conversation: '{_print_preview[:60]}{'...' if len(_print_preview) > 60 else ''}'")
    
    # 处理system prompt
    # ── System prompt (cached per session for prefix caching) ──
    # Built once on first call, reused for all subsequent calls.
    # Only rebuilt after context compression events (which invalidate
    # the cache and reload memory from disk).
    #
    # For continuing sessions (gateway creates a fresh AIAgent per
    # message), we load the stored system prompt from the session DB
    # instead of rebuilding.  Rebuilding would pick up memory changes
    # from disk that the model already knows about (it wrote them!),
    # producing a different system prompt and breaking the Anthropic
    # prefix cache.
    # ── System prompt（按 session 缓存，用于 prefix caching）──
    # 首次调用构建，后续复用；仅在上下文压缩后重建（会使缓存失效并重新加载 memory）
    #
    # 续接 session 时（gateway 每条消息新建 AIAgent），从 session DB 加载已存 system prompt
    # 而非重建——重建会读入磁盘上新 memory，与模型已知内容不一致，破坏 Anthropic prefix cache
    if self._cached_system_prompt is None:
        stored_prompt = None
        if conversation_history and self._session_db:
            try:
                session_row = self._session_db.get_session(self.session_id)
                if session_row:
                    stored_prompt = session_row.get("system_prompt") or None
            except Exception:
                pass  # Fall through to build fresh | 失败则 fall through 从零构建

        if stored_prompt:
            # Continuing session — reuse the exact system prompt from
            # the previous turn so the Anthropic cache prefix matches.
            # 续接 session — 复用上一回合完全相同的 system prompt，使 Anthropic cache prefix 匹配
            self._cached_system_prompt = stored_prompt
        else:
            # First turn of a new session — build from scratch.
            # 新 session 首回合 — 从零构建
            self._cached_system_prompt = self._build_system_prompt(system_message)
            # Plugin hook: on_session_start
            # Fired once when a brand-new session is created (not on
            # continuation).  Plugins can use this to initialise
            # session-scoped state (e.g. warm a memory cache).
            # 插件钩子：on_session_start
            # 仅在全新 session 创建时触发（非续接）；插件可初始化 session 级状态（如预热 memory cache）
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "on_session_start",
                    session_id=self.session_id,
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("on_session_start hook failed: %s", exc)

            # Store the system prompt snapshot in SQLite
            # 将 system prompt 快照存入 SQLite
            if self._session_db:
                try:
                    self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                except Exception as e:
                    logger.debug("Session DB update_system_prompt failed: %s", e)

    active_system_prompt = self._cached_system_prompt

    return messages,active_system_prompt

def preflight_compression():
        # ── Preflight context compression ──
    # Before entering the main loop, check if the loaded conversation
    # history already exceeds the model's context threshold.  This handles
    # cases where a user switches to a model with a smaller context window
    # while having a large existing session — compress proactively rather
    # than waiting for an API error (which might be caught as a non-retryable
    # 4xx and abort the request entirely).
    # ── 预检上下文压缩 ──
    # 进入主循环前检查已加载 history 是否超过模型上下文阈值
    # 处理用户切换到更小上下文窗口但 session 很大的情况——主动压缩而非等 API 4xx 直接失败
    if (
        self.compression_enabled
        and len(messages) > self.context_compressor.protect_first_n
                            + self.context_compressor.protect_last_n + 1
    ):
        # Include tool schema tokens — with many tools these can add
        # 20-30K+ tokens that the old sys+msg estimate missed entirely.
        # 估算时包含 tool schema token——工具多时可能额外 20-30K+，旧 sys+msg 估算会漏掉
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=self.tools or None,
        )

        if _preflight_tokens >= self.context_compressor.threshold_tokens:
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{self.context_compressor.threshold_tokens:,}",
                self.model,
                f"{self.context_compressor.context_length:,}",
            )
            if not self.quiet_mode:
                self._safe_print(
                    f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                    f">= {self.context_compressor.threshold_tokens:,} threshold"
                )
            # May need multiple passes for very large sessions with small
            # context windows (each pass summarises the middle N turns).
            # 超大 session + 小上下文可能需要多轮压缩（每轮摘要中间 N 个 turn）
            for _pass in range(3):
                _orig_len = len(messages)
                messages, active_system_prompt = self._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if len(messages) >= _orig_len:
                    break  # Cannot compress further | 无法再压缩
                # Compression created a new session — clear the history
                # reference so _flush_messages_to_session_db writes ALL
                # compressed messages to the new session's SQLite, not
                # skipping them because conversation_history is still the
                # pre-compression length.
                # 压缩创建了新 session — 清空 history 引用，使 _flush_messages_to_session_db
                # 把所有压缩后消息写入新 session SQLite，而非因 conversation_history 仍是压缩前长度而跳过
                conversation_history = None
                # Fix: reset retry counters after compression so the model
                # gets a fresh budget on the compressed context.  Without
                # this, pre-compression retries carry over and the model
                # hits "(empty)" immediately after compression-induced
                # context loss.
                # 修复：压缩后重置重试计数，让模型在压缩后上下文获得新预算
                # 否则压缩前重试会延续，模型压缩后立即返回 "(empty)"
                self._empty_content_retries = 0
                self._thinking_prefill_retries = 0
                self._last_content_with_tools = None
                self._last_content_tools_all_housekeeping = False
                self._mute_post_response = False
                # Re-estimate after compression
                # 压缩后重新估算 token
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=self.tools or None,
                )
                if _preflight_tokens < self.context_compressor.threshold_tokens:
                    break  # Under threshold | 低于阈值
    return None


    


def plugins_and_memory_prefetch():
    # Plugin hook: pre_llm_call
    # Fired once per turn before the tool-calling loop.  Plugins can
    # return a dict with a ``context`` key (or a plain string) whose
    # value is appended to the current turn's user message.
    #
    # Context is ALWAYS injected into the user message, never the
    # system prompt.  This preserves the prompt cache prefix — the
    # system prompt stays identical across turns so cached tokens
    # are reused.  The system prompt is Hermes's territory; plugins
    # contribute context alongside the user's input.
    #
    # All injected context is ephemeral (not persisted to session DB).
    # 插件钩子：pre_llm_call
    # 每回合工具循环前触发一次；插件可返回含 ``context`` 键的 dict（或纯字符串）
    # 追加到本回合 user 消息
    #
    # 上下文始终注入 user 消息，永不注入 system prompt——保持 prompt cache prefix 不变
    # system prompt 跨回合相同以复用 cached token；system prompt 归 Hermes，插件上下文与用户输入并列
    #
    # 所有注入上下文均为 ephemeral（不持久化到 session DB）
    _plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=self.session_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=self.model,
            platform=getattr(self, "platform", None) or "",
            sender_id=getattr(self, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        for r in _pre_results:
            if isinstance(r, dict) and r.get("context"):
                _ctx_parts.append(str(r["context"]))
            elif isinstance(r, str) and r.strip():
                _ctx_parts.append(r)
        if _ctx_parts:
            _plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    return None