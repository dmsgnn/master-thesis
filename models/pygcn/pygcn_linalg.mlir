#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {torch.debug_module_name = "GCN"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<2708x1433xf32>, %arg1: tensor<2708x2708xf32>) -> tensor<2708x7xf32> {
    %cst = arith.constant dense<[0.119321413, 0.116671227, -0.438798875, -0.598985076, 0.145210594, -0.00804764777, 1.00821221]> : tensor<7xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<16x7xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<16xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1433x16xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_4 = arith.constant -3.40282347E+38 : f32
    %0 = tensor.empty() : tensor<2708x16xf32>
    %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<2708x16xf32>) -> tensor<2708x16xf32>
    %2 = linalg.matmul ins(%arg0, %cst_2 : tensor<2708x1433xf32>, tensor<1433x16xf32>) outs(%1 : tensor<2708x16xf32>) -> tensor<2708x16xf32>
    %3 = linalg.matmul ins(%arg1, %2 : tensor<2708x2708xf32>, tensor<2708x16xf32>) outs(%1 : tensor<2708x16xf32>) -> tensor<2708x16xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_1 : tensor<2708x16xf32>, tensor<16xf32>) outs(%0 : tensor<2708x16xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.addf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<2708x16xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<2708x16xf32>) outs(%0 : tensor<2708x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = arith.cmpf ugt, %in, %cst_3 : f32
      %23 = arith.select %22, %in, %cst_3 : f32
      linalg.yield %23 : f32
    } -> tensor<2708x16xf32>
    %6 = tensor.empty() : tensor<2708x7xf32>
    %7 = linalg.fill ins(%cst_3 : f32) outs(%6 : tensor<2708x7xf32>) -> tensor<2708x7xf32>
    %8 = linalg.matmul ins(%5, %cst_0 : tensor<2708x16xf32>, tensor<16x7xf32>) outs(%7 : tensor<2708x7xf32>) -> tensor<2708x7xf32>
    %9 = linalg.matmul ins(%arg1, %8 : tensor<2708x2708xf32>, tensor<2708x7xf32>) outs(%7 : tensor<2708x7xf32>) -> tensor<2708x7xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %cst : tensor<2708x7xf32>, tensor<7xf32>) outs(%6 : tensor<2708x7xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.addf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<2708x7xf32>
    %11 = tensor.empty() : tensor<2708x1xi64>
    %12 = linalg.fill ins(%c0_i64 : i64) outs(%11 : tensor<2708x1xi64>) -> tensor<2708x1xi64>
    %13 = tensor.empty() : tensor<2708x1xf32>
    %14 = linalg.fill ins(%cst_4 : f32) outs(%13 : tensor<2708x1xf32>) -> tensor<2708x1xf32>
    %15:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<2708x7xf32>) outs(%14, %12 : tensor<2708x1xf32>, tensor<2708x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_5: i64):
      %22 = linalg.index 1 : index
      %23 = arith.index_cast %22 : index to i64
      %24 = arith.maxf %in, %out : f32
      %25 = arith.cmpf ogt, %in, %out : f32
      %26 = arith.select %25, %23, %out_5 : i64
      linalg.yield %24, %26 : f32, i64
    } -> (tensor<2708x1xf32>, tensor<2708x1xi64>)
    %16 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %15#0 : tensor<2708x7xf32>, tensor<2708x1xf32>) outs(%6 : tensor<2708x7xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.subf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<2708x7xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<2708x7xf32>) outs(%6 : tensor<2708x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.exp %in : f32
      linalg.yield %22 : f32
    } -> tensor<2708x7xf32>
    %18 = linalg.fill ins(%cst_3 : f32) outs(%13 : tensor<2708x1xf32>) -> tensor<2708x1xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%17 : tensor<2708x7xf32>) outs(%18 : tensor<2708x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = arith.addf %in, %out : f32
      linalg.yield %22 : f32
    } -> tensor<2708x1xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<2708x1xf32>) outs(%13 : tensor<2708x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.log %in : f32
      linalg.yield %22 : f32
    } -> tensor<2708x1xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%16, %20 : tensor<2708x7xf32>, tensor<2708x1xf32>) outs(%6 : tensor<2708x7xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.subf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<2708x7xf32>
    return %21 : tensor<2708x7xf32>
  }
}
