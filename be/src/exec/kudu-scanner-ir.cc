// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "exec/kudu-scanner.h"
#include "runtime/runtime-state.h"
#include "runtime/row-batch.h"
#include "runtime/tuple-row.h"

using namespace impala;

Status KuduScanner::DecodeRowsIntoRowBatch(RowBatch* row_batch, Tuple** tuple_mem) {
  // Short-circuit the count(*) case.
  if (scan_node_->tuple_desc()->slots().empty()) {
    return HandleEmptyProjection(row_batch);
  }

  for (int krow_idx = cur_kudu_batch_num_read_; krow_idx < num_rows_; ++krow_idx) {
    Tuple* kudu_tuple = const_cast<Tuple*>(reinterpret_cast<const Tuple*>(data_ + krow_idx * row_size_));
    ++cur_kudu_batch_num_read_;

    // Kudu tuples containing TIMESTAMP columns (UNIXTIME_MICROS in Kudu, stored as an
    // int64) have 8 bytes of padding following the timestamp. Because this padding is
    // provided, Impala can convert these unixtime values to Impala's TimestampValue
    // format in place and copy the rows to Impala row batches.
    // TODO: avoid mem copies with a Kudu mem 'release' mechanism, attaching mem to the
    // batch.
    // TODO: consider codegen for this per-timestamp col fixup
    /*for (const SlotDescriptor* slot : timestamp_slots_) {
      DCHECK(slot->type().type == TYPE_TIMESTAMP);
      if (slot->is_nullable() && kudu_tuple->IsNull(slot->null_indicator_offset())) {
        continue;
      }
      int64_t ts_micros = *reinterpret_cast<int64_t*>(
          kudu_tuple->GetSlot(slot->tuple_offset()));
      int64_t ts_seconds = ts_micros / MICROS_PER_SEC;
      int64_t micros_part = ts_micros - (ts_seconds * MICROS_PER_SEC);
      TimestampValue tv = TimestampValue::FromUnixTimeMicros(ts_seconds, micros_part);
      if (tv.HasDateAndTime()) {
        RawValue::Write(&tv, kudu_tuple, slot, NULL);
      } else {
        kudu_tuple->SetNull(slot->null_indicator_offset());
        RETURN_IF_ERROR(state_->LogOrReturnError(
            ErrorMsg::Init(TErrorCode::KUDU_TIMESTAMP_OUT_OF_RANGE,
              scan_node_->table_->name(),
              scan_node_->table_->schema().Column(slot->col_pos()).name())));
      }
    }*/

    // Evaluate runtime filters that haven't been pushed down to Kudu.
    /*if (!EvalRuntimeFilters(reinterpret_cast<TupleRow*>(output_row))) {
        continue;
    }*/

    // Evaluate the conjuncts that haven't been pushed down to Kudu. Conjunct evaluation
    // is performed directly on the Kudu tuple because its memory layout is identical to
    // Impala's. We only copy the surviving tuples to Impala's output row batch.
    if (!conjunct_ctxs_.empty() && !ExecNode::EvalConjuncts(&conjunct_ctxs_[0],
        conjunct_ctxs_.size(), reinterpret_cast<TupleRow*>(&kudu_tuple))) {
      continue;
    }
    // Deep copy the tuple, set it in a new row, and commit the row.
    kudu_tuple->DeepCopy(*tuple_mem, *scan_node_->tuple_desc(),
        row_batch->tuple_data_pool());
    TupleRow* row = row_batch->GetRow(row_batch->AddRow());
    row->SetTuple(0, *tuple_mem);
    row_batch->CommitLastRow();
    // If we've reached the capacity, or the LIMIT for the scan, return.
    if (row_batch->AtCapacity() || scan_node_->ReachedLimit()) break;
    // Move to the next tuple in the tuple buffer.
    *tuple_mem = next_tuple(*tuple_mem);
  }
  ExprContext::FreeLocalAllocations(conjunct_ctxs_);

  // Check the status in case an error status was set during conjunct evaluation.
  return state_->GetQueryStatus();
}
