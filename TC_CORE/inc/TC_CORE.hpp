﻿/*
 * Copyright 2019 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tc_core_export.h" // generated by CMake
#include "tc_core_version.h"
#include <cstdint>
#include <utility>

namespace VPF
{

/* Interface for data exchange;
 * It represents memory object (CPU- or GPU-side memory etc.);
 * token 是数据交换的接口，
 * buffer cudabuffer surface都是继承这个类
 * 实现了内存的分配和管理
 */
class TC_CORE_EXPORT Token
{
public:
  Token& operator=(const Token& other) = delete;
  Token(const Token& other) = delete;

  virtual ~Token();

protected:
  Token();
};

// 任务执行状态 成功 失败
enum class TaskExecStatus { TASK_EXEC_SUCCESS, TASK_EXEC_FAIL };

/* Synchronization call which will be done after a blocking task;
 */
typedef void (*p_sync_call)(void* p_args);

/* Task is unit of processing; Inherit from this class to add user-defined
 * processing stage;
 * 任务是处理的基本单元；从这个类继承以添加用户自定义的处理
 *
 */
class TC_CORE_EXPORT Task
{
public:
  Task() = delete;
  Task(const Task& other) = delete;
  Task& operator=(const Task& other) = delete;

  virtual ~Task();

  /* Method to be overridden in ancestors;
   * 核心代码 子类实现
   */
  virtual TaskExecStatus Run();

  /* Call this method to run the task;
   * 调用此方法执行任务
   * 里面调用run函数
   */
  virtual TaskExecStatus Execute();

  /* Sets given token as input;
   * Doesn't take ownership of object passed by pointer, only stores it
   * within inplementation;
   * 设置给定的令牌作为输入；不获取通过指针传递对象的所有权，仅在实现中存储它
   */
  bool SetInput(Token* input, uint32_t input_num);

  /* Sets given token as output;
   * Doesn't take ownership of object passed by pointer, only stores it
   * within inplementation;
   * 设置输出的内存
   */
  bool SetOutput(Token* output, uint32_t output_num);

  /* Sets all inputs to nullptr;
   */
  void ClearInputs();

  /* Sets all outputs to nullptr;
   */
  void ClearOutputs();

  /* Returns pointer to task input in case of success, nullptr otherwise;
   */
  Token* GetInput(uint32_t num_input = 0);

  /* Returns pointer to task output in case of success, nullptr otherwise;
   */
  Token* GetOutput(uint32_t num_output = 0);

  /* Returns number of outputs;
   */
  uint64_t GetNumOutputs() const;

  /* Returns number of inputs;
   */
  uint64_t GetNumInputs() const;

  /* Returns task name;
   */
  const char* GetName() const;

protected:
  Task(const char* str_name, uint32_t num_inputs, uint32_t num_outputs,
       p_sync_call sync_call = nullptr, void* p_args = nullptr);

  /* Hidden implementation;
   * IMPL实现，
   */
  struct TaskImpl* p_impl = nullptr;
};
} // namespace VPF
