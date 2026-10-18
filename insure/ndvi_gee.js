// =====================================================================================================
// SATELLITE CROP GREENNESS (NDVI) FOR EVERY INSURED DISTRICT — Google Earth Engine script
//
// Why: weather alone explains little of district harvests. NDVI (how green the vegetation is, from the MODIS
// satellites every 16 days at 250 m) sees what rain data cannot: pests, disease, poor planting, late rains.
//
// How (free for research / non-commercial use):
//  1. Sign up at https://earthengine.google.com (free non-commercial account) and open https://code.earthengine.google.com
//  2. Download the district boundaries: https://github.com/HarvestStat/HarvestStat-Africa/raw/main/public/hvstat_africa_boundary_v1.2_shp.zip
//  3. In the Code Editor: Assets > NEW > Shape files > choose that zip > name it  hvstat_boundary  > UPLOAD (wait for it in Tasks)
//  4. Paste this script, set ASSET below to your asset path, press Run, then in the Tasks tab press RUN on "ndvi_districts".
//     It exports a CSV (fnid, date, ndvi) to your Google Drive. It can take a few hours: Earth Engine runs it in the cloud.
//  5. Put the CSV in the project as  data/ndvi_districts.csv  and retrain insurance:
//        sudo docker compose exec web python insure/train.py
//     The model then uses each season's greenness compared with the district's normal (see insure/ndvi.py).
// =====================================================================================================
var ASSET = 'projects/YOUR_PROJECT/assets/hvstat_boundary';   // <-- your asset path (Assets panel, "copy path")
var ID_FIELD = 'fnid';                                        // district id field (open the asset to check: fnid or FNID)

var districts = ee.FeatureCollection(ASSET).select([ID_FIELD]);
var cropland = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40);          // ESA WorldCover 2021: class 40 = cropland
var ndvi = ee.ImageCollection('MODIS/061/MOD13Q1')                                // 16-day composites, 250 m, since 2000
  .filterDate('2001-01-01', '2026-01-01')
  .select('NDVI');

var table = ndvi.map(function (img) {
  var v = img.multiply(0.0001).updateMask(cropland);                               // scale to -1..1, cropland pixels only
  var stats = v.reduceRegions({collection: districts, reducer: ee.Reducer.mean(), scale: 1000, tileScale: 4});
  var date = img.date().format('YYYY-MM-dd');
  return stats.map(function (f) { return ee.Feature(null, {fnid: f.get(ID_FIELD), date: date, ndvi: f.get('mean')}); });
}).flatten().filter(ee.Filter.notNull(['ndvi']));

print('first rows', table.limit(5));
Export.table.toDrive({collection: table, description: 'ndvi_districts', fileFormat: 'CSV', selectors: ['fnid', 'date', 'ndvi']});
