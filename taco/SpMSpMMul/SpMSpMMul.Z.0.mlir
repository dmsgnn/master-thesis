#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @SpMSpMMul.Z.0.main(%arg0: tensor<15x15xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>, %arg1: tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], dimToLvl = affine_map<(d0, d1) -> (d1, d0)> }>>) -> tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>> attributes {llvm.emit_c_interface} {
    %0 = bufferization.alloc_tensor() : tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<15x15xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>, tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ], dimToLvl = affine_map<(d0, d1) -> (d1, d0)> }>>) outs(%0 : tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>
    return %1 : tensor<15x16xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>
  }
}

