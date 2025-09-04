import { app } from "/scripts/app.js";

app.registerExtension({
	name: "NanoBanana.AICG.SplitNodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// 支持两个分离的节点
		if (nodeData.name === "NanoBananaTextToImage" || nodeData.name === "NanoBananaImageToImage") {
			
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				// 初始化倒计时状态
				this.queryCooldown = 0;
				this.cooldownInterval = null;

				// 获取节点类型用于显示
				const nodeTypeName = nodeData.name === "NanoBananaTextToImage" ? "文生图" : "图生图";

				// 创建购买次数按钮（放在右上角）
				const purchaseButton = this.addWidget("button", "购买次数请点击这里", "purchase", () => {
					// 点击时跳转到购买页面
					window.open("https://example.com/purchase", "_blank");
				});

				// 设置购买按钮的样式（右对齐）
				if (purchaseButton.element) {
					purchaseButton.element.style.float = "right";
					purchaseButton.element.style.marginRight = "10px";
				}

				// 创建"查询"按钮
				const queryButton = this.addWidget("button", "查询次数", "query", () => {
					this.verifyAuthCode(); // 点击时调用验证函数
				});

				// 创建用于显示结果的文本区域
				const resultWidget = this.addWidget("text", `${nodeTypeName}查询结果:`, "点击上方按钮查询授权码状态", { 
                    multiline: true,
                });

				// 将组件保存到节点实例上，方便后续访问
				this.queryButton = queryButton;
				this.resultWidget = resultWidget;
				this.purchaseButton = purchaseButton;
				this.nodeTypeName = nodeTypeName; // 保存节点类型名称

				// 启动倒计时显示
				this.startCooldownDisplay = () => {
					if (this.cooldownInterval) {
						clearInterval(this.cooldownInterval);
					}

					this.cooldownInterval = setInterval(() => {
						if (this.queryCooldown > 0) {
							this.queryCooldown--;
							this.queryButton.name = `查询次数 (${this.queryCooldown}s)`;
							this.queryButton.disabled = true;
						} else {
							this.queryButton.name = "查询次数";
							this.queryButton.disabled = false;
							if (this.cooldownInterval) {
								clearInterval(this.cooldownInterval);
								this.cooldownInterval = null;
							}
						}
					}, 1000);
				};

				// 定义按钮点击时触发的查询函数
				this.verifyAuthCode = async () => {
					const authCodeWidget = this.widgets.find(w => w.name === "授权码");
					if (!authCodeWidget) { 
						this.resultWidget.value = "错误：找不到授权码输入框。";
						return; 
					}
					
					const authCode = authCodeWidget.value?.trim();
					if (!authCode) {
						this.resultWidget.value = "❌ 错误：请输入授权码后再查询。";
						return;
					}

					// 检查是否在冷却时间内
					if (this.queryCooldown > 0) {
						this.resultWidget.value = `⏱️ 查询太频繁，请等待 ${this.queryCooldown} 秒后再试。`;
						return;
					}

					this.resultWidget.value = "🔍 正在查询中...";
					this.queryButton.disabled = true;
					this.queryButton.name = "查询中...";

					try {
						// 调用后端的API接口
						const response = await fetch("/nanobanana/verify", {
							method: "POST",
							headers: { "Content-Type": "application/json" },
							body: JSON.stringify({ auth_code: authCode }),
						});

						if (!response.ok) {
							throw new Error(`HTTP ${response.status}: ${response.statusText}`);
						}

						const data = await response.json();

						if (data.success) {
							// 查询成功
							let statusInfo = "";
							if (data.status) {
								statusInfo += `\n状态: ${data.status}`;
							}
							if (data.usage_percent) {
								statusInfo += `\n使用率: ${data.usage_percent}%`;
							}
							if (data.expire_info) {
								statusInfo += `\n有效期: ${data.expire_info}`;
							}

							this.resultWidget.value = `✅ ${this.nodeTypeName}查询成功!\n剩余次数: ${data.remaining}${statusInfo}`;
							
							// 设置30秒冷却时间
							this.queryCooldown = 30;
							this.startCooldownDisplay();

						} else {
							// 查询失败
							if (data.cooldown && data.cooldown > 0) {
								// 如果是频率限制导致的失败，设置相应的冷却时间
								this.queryCooldown = data.cooldown;
								this.startCooldownDisplay();
								this.resultWidget.value = `⏱️ 查询太频繁!\n请等待 ${data.cooldown} 秒后再试`;
							} else {
								this.resultWidget.value = `❌ ${this.nodeTypeName}查询失败!\n原因: ${data.error}`;
								// 即使失败也设置冷却时间，防止频繁请求
								this.queryCooldown = 10;
								this.startCooldownDisplay();
							}
						}
					} catch (error) {
						console.error("查询请求失败:", error);
						this.resultWidget.value = `❌ ${this.nodeTypeName}网络请求失败:\n${error.message}`;
						
						// 网络错误也设置短暂冷却时间
						this.queryCooldown = 5;
						this.startCooldownDisplay();
					} finally {
						// 如果不在冷却状态，重置按钮
						if (this.queryCooldown <= 0) {
							this.queryButton.disabled = false;
							this.queryButton.name = "查询次数";
						}
					}
				};

				// 节点销毁时清理定时器
				const onRemoved = this.onRemoved;
				this.onRemoved = function() {
					if (this.cooldownInterval) {
						clearInterval(this.cooldownInterval);
						this.cooldownInterval = null;
					}
					if (onRemoved) {
						onRemoved.apply(this, arguments);
					}
				};
			};

			// 添加生成后自动更新的逻辑
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function(message) {
				onExecuted?.apply(this, arguments);

                const statusText = message?.text?.[0] ?? (Array.isArray(message) ? message[1] : null);

				if (statusText && this.resultWidget) {
					// 检查是否包含剩余次数信息
					const remainingMatch = statusText.match(/剩余次数:\s*(\d+)/);
					if (remainingMatch && remainingMatch[1]) {
						const newRemaining = remainingMatch[1];
						
						// 检查是否生成成功
						if (statusText.includes('❌') || statusText.includes('⚠️')) {
							// 生成失败
							this.resultWidget.value = `❌ ${this.nodeTypeName}生成失败!\n最新剩余次数: ${newRemaining}\n原因: ${statusText}`;
						} else {
							// 生成成功
							this.resultWidget.value = `✅ ${this.nodeTypeName}生成成功!\n最新剩余次数: ${newRemaining}`;
						}
					} else {
						// 没有剩余次数信息，可能是错误消息
						this.resultWidget.value = `ℹ️ ${this.nodeTypeName}执行结果:\n${statusText}`;
					}
				}
			};
		}
	},
});
