/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_TRANSFER_SERVER_LINUX_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_TRANSFER_SERVER_LINUX_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/transfer_server_interface.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/socket-server.h"
#include "xla/python/transfer/streaming_ifrt.h"

namespace xla {
namespace ifrt {

class PjRtTransferServerLinux : public TransferServerInterface {
 public:
  // Starts the DCN SocketServer if it is not already started.
  absl::Status StartTransferServerIfNotStarted(
      std::shared_ptr<xla::PjRtClient> pjrt_client,
      std::shared_ptr<xla::KeyValueStoreInterface> kv_store, int process_index)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(socket_server_mu_) override;

  // Awaits a pull from a remote process.
  absl::Status CrossHostAwaitPull(int64_t uuid,
                                  absl::Span<xla::ifrt::ArrayRef> arrays,
                                  const std::vector<int>& buffer_idxs) override;

  // Pulls buffers from a remote process.
  absl::Status CrossHostPull(
      std::shared_ptr<xla::PjRtClient> pjrt_client,
      std::shared_ptr<xla::KeyValueStoreInterface> kv_store, int64_t uuid,
      absl::Span<xla::ifrt::ArrayRef> arrays, std::vector<int>& dst_device_idxs,
      xla::ifrt::DeviceListRef dst_devices,
      std::optional<MemoryKind> memory_kind, int remote_pid,
      absl::btree_map<int, PjRtArray::PjRtBuffers>& buffer_list) override;

  static absl::StatusOr<std::unique_ptr<PjRtTransferServerLinux>> Create(
      xla::ifrt::PjRtClient::CreateOptions& options);

  PjRtTransferServerLinux(size_t transfer_size, size_t max_num_parallel_copies,
                          absl::Duration cross_host_transfer_timeout,
                          aux::SocketAddress socket_address,
                          std::vector<aux::SocketAddress> transport_addresses);

 private:
  size_t transfer_size_;
  size_t max_num_parallel_copies_;
  absl::Duration cross_host_transfer_timeout_;
  aux::SocketAddress socket_address_;
  std::vector<aux::SocketAddress> transport_addresses_;
  std::optional<std::shared_ptr<aux::SocketServer>> socket_server_;
  std::optional<std::shared_ptr<aux::PremappedCopierState>> premapped_copier_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_TRANSFER_SERVER_LINUX_H_
