/* Copyright (c) 2008-2026 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "header.h"
#include "image.h"

#include "math/math.h"
#include "misc/bitset.h"

#include "algo/loop.h"
#include "interp/linear.h"
#include "interp/nearest.h"

#include "fixel/legacy/fixel_metric.h"
#include "fixel/legacy/image.h"
#include "fixel/legacy/keys.h"

#include "dwi/tractography/file.h"
#include "dwi/tractography/streamline.h"

#include "dwi/tractography/SIFT2/coeff_optimiser.h"
#include "dwi/tractography/SIFT2/fixel_updater.h"
#include "dwi/tractography/SIFT2/reg_calculator.h"
#include "dwi/tractography/SIFT2/streamline_stats.h"
#include "dwi/tractography/SIFT2/tckfactor.h"

#include "dwi/tractography/SIFT/track_index_range.h"






namespace MR {
  namespace DWI {
    namespace Tractography {
      namespace SIFT2 {




      void TckFactor::set_reg_lambdas (const double lambda_tikhonov, const double lambda_tv)
      {
        assert (num_tracks());
        double A = 0.0;
        for (size_t i = 1; i != fixels.size(); ++i)
          A += fixels[i].get_weight() * Math::pow2 (fixels[i].get_FOD());

        A /= double(num_tracks());
        INFO ("Constant A scaling regularisation terms to match data term is " + str(A));
        reg_multiplier_tikhonov = lambda_tikhonov * A;
        reg_multiplier_tv       = lambda_tv * A;
      }



      void TckFactor::load_microstructure_map (const std::string& map_path, const std::string& tracks_path,
                                               const std::string& parcel_path, const std::string& classes_path)
      {
        assert (num_tracks());

        auto micro_image = Image<float>::open (map_path);
        const bool is_3d = (micro_image.ndim() == 3) ||
                           (micro_image.ndim() == 4 && micro_image.size(3) == 1);
        if (!is_3d)
          throw Exception ("Microstructure map must be a 3D image (got " + str(micro_image.ndim()) + "D with size " +
                           str(micro_image.size(0)) + "x" + str(micro_image.size(1)) + "x" + str(micro_image.size(2)) +
                           (micro_image.ndim() > 3 ? ("x" + str(micro_image.size(3))) : "") + ")");

        const double vox_x = micro_image.spacing (0);
        const double vox_y = micro_image.spacing (1);
        const double vox_z = micro_image.spacing (2);
        const double sampling_interval = 0.5 * std::min ({vox_x, vox_y, vox_z});

        INFO ("Microstructure map voxel sizes: " + str(vox_x) + " x " + str(vox_y) + " x " + str(vox_z) +
              " mm; sampling interval: " + str(sampling_interval) + " mm");

        Interp::Linear<decltype(micro_image)> micro_interp (micro_image);

        // Optional parcellation setup
        const bool do_parcellation = (!parcel_path.empty() && !classes_path.empty());
        std::map<int, bool> label_is_subcortical;
        std::unique_ptr<Image<int32_t>> parcel_image_ptr;
        std::unique_ptr<Interp::Nearest<Image<int32_t>>> parcel_interp_ptr;

        if (do_parcellation) {
          std::ifstream csv_file (classes_path);
          if (!csv_file.is_open())
            throw Exception ("Unable to open parcellation classes CSV: \"" + classes_path + "\"");

          std::string line;
          bool header_skipped = false;
          while (std::getline (csv_file, line)) {
            if (line.empty())
              continue;

            if (!header_skipped) {
              if (line.find_first_of ("0123456789") != 0) {
                header_skipped = true;
                continue;
              }
              header_skipped = true;
            }

            std::istringstream ss (line);
            std::string intensity_str, class_str;
            if (!std::getline (ss, intensity_str, ',') || !std::getline (ss, class_str, ','))
              throw Exception ("Malformed line in parcellation classes CSV: \"" + line + "\"");

            const size_t start = class_str.find_first_not_of (" \t\r\n");
            const size_t end = class_str.find_last_not_of (" \t\r\n");
            if (start == std::string::npos)
              throw Exception ("Empty class in parcellation classes CSV: \"" + line + "\"");
            class_str = class_str.substr (start, end - start + 1);

            const int intensity = std::stoi (intensity_str);

            if (class_str == "Subcortical" || class_str == "subcortical")
              label_is_subcortical[intensity] = true;
            else if (class_str == "Cortical" || class_str == "cortical")
              label_is_subcortical[intensity] = false;
            else
              throw Exception ("Unknown class \"" + class_str + "\" in parcellation CSV (expected Subcortical or Cortical)");
          }

          INFO (str(label_is_subcortical.size()) + " region labels loaded from parcellation classes CSV:");
          for (const auto& kv : label_is_subcortical)
            INFO ("  Label " + str(kv.first) + ": " + std::string(kv.second ? "Subcortical" : "Cortical"));

          parcel_image_ptr.reset (new Image<int32_t> (Image<int32_t>::open (parcel_path)));
          parcel_interp_ptr.reset (new Interp::Nearest<Image<int32_t>> (*parcel_image_ptr));

          // Scan parcellation image for label voxel counts
          {
            std::map<int, size_t> label_voxel_count;
            auto parcel_scan = *parcel_image_ptr;
            for (auto l = Loop (parcel_scan) (parcel_scan); l; ++l) {
              const int label = parcel_scan.value();
              if (label != 0)
                ++label_voxel_count[label];
            }
            INFO (str(label_voxel_count.size()) + " unique non-zero labels found in parcellation image:");
            for (const auto& kv : label_voxel_count) {
              const auto it = label_is_subcortical.find (kv.first);
              const std::string cls = (it != label_is_subcortical.end())
                  ? (it->second ? "Subcortical" : "Cortical")
                  : "NOT IN CSV";
              INFO ("  Label " + str(kv.first) + ": " + str(kv.second) + " voxels (" + cls + ")");
            }
            size_t csv_missing = 0;
            for (const auto& kv : label_is_subcortical) {
              if (label_voxel_count.find (kv.first) == label_voxel_count.end()) {
                if (csv_missing == 0)
                  WARN ("The following CSV labels are not present in the parcellation image (typo or wrong file?):");
                WARN ("  Label " + str(kv.first) + " (" + std::string(kv.second ? "Subcortical" : "Cortical") + ")");
                ++csv_missing;
              }
            }
            if (!csv_missing)
              INFO ("All CSV labels are present in the parcellation image.");
          }
        }

        microstructure_af.resize (num_tracks());
        micro_blend.resize (num_tracks());
        micro_blend.setOnes();

        size_t clamped_count = 0;
        size_t no_sample_count = 0;
        size_t count_sub_sub = 0, count_sub_cor = 0, count_cor_cor = 0, count_unknown = 0;
        std::map<int, size_t> endpoint_label_counts;
        size_t endpoints_outside_fov = 0;

        {
          Tractography::Properties properties;
          Tractography::Reader<float> reader (tracks_path, properties);
          Tractography::Streamline<float> tck;
          ProgressBar progress ("Sampling microstructure map along streamlines", num_tracks());

          SIFT::track_t track_index = 0;
          while (reader (tck) && track_index < num_tracks()) {

            // --- Microstructure sampling ---
            double sum = 0.0;
            size_t sample_count = 0;

            if (tck.size() >= 2) {
              double dist_along_segment = 0.0;
              size_t point_idx = 0;

              while (point_idx < tck.size() - 1) {
                const Eigen::Vector3f& p0 = tck[point_idx];
                const Eigen::Vector3f& p1 = tck[point_idx + 1];
                const Eigen::Vector3f segment = p1 - p0;
                const double segment_length = segment.norm();

                if (segment_length < 1e-12) {
                  ++point_idx;
                  dist_along_segment = 0.0;
                  continue;
                }

                while (dist_along_segment <= segment_length) {
                  const double frac = dist_along_segment / segment_length;
                  const Eigen::Vector3f sample_point = p0 + frac * segment;

                  if (micro_interp.scanner (sample_point)) {
                    const float val = micro_interp.value();
                    if (std::isfinite (val) && val > 0.0f) {
                      sum += val;
                      ++sample_count;
                    }
                  }

                  dist_along_segment += sampling_interval;
                }

                dist_along_segment -= segment_length;
                ++point_idx;
              }
            }

            if (sample_count > 0) {
              const double mean = sum / sample_count;

              if (mean < SIFT2_MICRO_AF_EPSILON) {
                microstructure_af[track_index] = SIFT2_MICRO_AF_EPSILON;
                ++clamped_count;
              } else {
                microstructure_af[track_index] = mean;
              }
            } else {
              microstructure_af[track_index] = 1.0;
              ++no_sample_count;
            }

            // --- Parcellation-based blend classification ---
            if (do_parcellation) {
              if (tck.size() < 2) {
                micro_blend[track_index] = 0.0;
                ++count_unknown;
              } else {
                int label_start = 0, label_end = 0;
                if (parcel_interp_ptr->scanner (tck.front())) {
                  label_start = parcel_interp_ptr->value();
                  ++endpoint_label_counts[label_start];
                } else {
                  ++endpoints_outside_fov;
                }
                if (parcel_interp_ptr->scanner (tck.back())) {
                  label_end = parcel_interp_ptr->value();
                  ++endpoint_label_counts[label_end];
                } else {
                  ++endpoints_outside_fov;
                }

                auto it_start = label_is_subcortical.find (label_start);
                auto it_end   = label_is_subcortical.find (label_end);

                const bool start_known = (it_start != label_is_subcortical.end());
                const bool end_known   = (it_end   != label_is_subcortical.end());

                if (start_known && end_known) {
                  const bool start_sub = it_start->second;
                  const bool end_sub   = it_end->second;

                  if (start_sub && end_sub) {
                    micro_blend[track_index] = 1.0;
                    ++count_sub_sub;
                  } else if (start_sub || end_sub) {
                    micro_blend[track_index] = 0.5;
                    ++count_sub_cor;
                  } else {
                    micro_blend[track_index] = 0.0;
                    ++count_cor_cor;
                  }
                } else {
                  micro_blend[track_index] = 0.0;
                  ++count_unknown;
                }
              }
            }

            ++track_index;
            ++progress;
          }

          if (track_index != num_tracks())
            throw Exception ("Track file contains " + str(track_index) + " streamlines but expected " + str(num_tracks()));
        }

        if (clamped_count)
          WARN (str(clamped_count) + " streamlines had MicroAF below " + str(SIFT2_MICRO_AF_EPSILON) + " and were clamped");
        if (no_sample_count)
          WARN (str(no_sample_count) + " streamlines had no valid samples from the microstructure map; set to neutral (1.0)");

        if (do_parcellation) {
          INFO ("Parcellation endpoint classification:");
          INFO ("  Subcortical-Subcortical (blend=1.0): " + str(count_sub_sub) + " streamlines");
          INFO ("  Subcortical-Cortical    (blend=0.5): " + str(count_sub_cor) + " streamlines");
          INFO ("  Cortical-Cortical       (blend=0.0): " + str(count_cor_cor) + " streamlines");
          if (count_unknown)
            WARN ("  Unknown/unclassified    (blend=0.0): " + str(count_unknown) + " streamlines");
          if (endpoints_outside_fov)
            WARN ("  " + str(endpoints_outside_fov) + " endpoints landed outside the parcellation image FOV");

          // Report endpoint label distribution (top 30 most frequent)
          INFO ("Endpoint label hit distribution (" + str(endpoint_label_counts.size()) + " unique labels hit across all endpoints):");
          std::vector<std::pair<size_t, int>> sorted_endpoints;
          sorted_endpoints.reserve (endpoint_label_counts.size());
          for (const auto& kv : endpoint_label_counts)
            sorted_endpoints.push_back ({kv.second, kv.first});
          std::sort (sorted_endpoints.rbegin(), sorted_endpoints.rend());
          const size_t max_print = std::min (sorted_endpoints.size(), size_t(30));
          for (size_t i = 0; i < max_print; ++i) {
            const int lbl    = sorted_endpoints[i].second;
            const size_t cnt = sorted_endpoints[i].first;
            const auto it    = label_is_subcortical.find (lbl);
            const std::string cls = (it != label_is_subcortical.end())
                ? (it->second ? "Subcortical" : "Cortical")
                : "NOT IN CSV";
            INFO ("  Label " + str(lbl) + ": " + str(cnt) + " endpoints (" + cls + ")");
          }
          if (sorted_endpoints.size() > max_print)
            INFO ("  ... (" + str(sorted_endpoints.size() - max_print) + " further labels omitted)");
        }

        has_microstructure = true;

        INFO ("Microstructure map loaded from \"" + map_path + "\"");
      }



      void TckFactor::apply_micro_strength (const double alpha)
      {
        if (!has_microstructure)
          throw Exception ("apply_micro_strength() called but no microstructure map was loaded");
        if (alpha <= 0.0)
          return;

        size_t n_sub_sub = 0, n_sub_cor = 0, n_unaffected = 0;
        for (SIFT::track_t i = 0; i != num_tracks(); ++i) {
          const double b = micro_blend[i];
          if (b <= 0.0) {
            ++n_unaffected;
            continue;
          }
          const double effective_alpha = alpha * b;
          coefficients[i] = (1.0 - effective_alpha) * coefficients[i]
                           + effective_alpha * std::log (microstructure_af[i]);
          if (b > 0.5 + 1e-6)
            ++n_sub_sub;
          else
            ++n_sub_cor;
        }

        INFO ("Applied -micro_strength " + str(alpha) + ":");
        INFO ("  Sub-Sub (blend=" + str(alpha)       + "): " + str(n_sub_sub)    + " streamlines");
        INFO ("  Sub-Cor (blend=" + str(alpha * 0.5) + "): " + str(n_sub_cor)    + " streamlines");
        INFO ("  Other   (blend=0.0): "               + str(n_unaffected) + " streamlines unchanged");
      }



      void TckFactor::store_orig_TDs()
      {
        for (vector<Fixel>::iterator i = fixels.begin(); i != fixels.end(); ++i)
          i->store_orig_TD();
      }



      void TckFactor::remove_excluded_fixels (const float min_td_frac)
      {
        Model<Fixel>::remove_excluded_fixels();

        // In addition to the complete exclusion, want to identify poorly-tracked fixels and
        //   exclude them from the optimisation (though they will still remain in the model)

        // There's no particular pattern to it; just use a hard threshold
        // Would prefer not to actually modify the streamline visitations; just exclude fixels from optimisation
        const double fixed_mu = mu();
        const double cf = calc_cost_function();
        SIFT::track_t excluded_count = 0, zero_TD_count = 0;
        double zero_TD_cf_sum = 0.0, excluded_cf_sum = 0.0;
        vector<Fixel>::iterator i = fixels.begin(); // SKip first fixel, which is an intentional null in DWI::Fixel_map<>
        for (++i; i != fixels.end(); ++i) {
          if (!i->get_orig_TD()) {
            ++zero_TD_count;
            zero_TD_cf_sum += i->get_cost (fixed_mu);
          } else if ((fixed_mu * i->get_orig_TD() < min_td_frac * i->get_FOD()) || (i->get_count() == 1)) {
            i->exclude();
            ++excluded_count;
            excluded_cf_sum += i->get_cost (fixed_mu);
          }
        }
        INFO (str(zero_TD_count) + " fixels have no attributed streamlines; these account for " + str(100.0 * zero_TD_cf_sum / cf) + "\% of the initial cost function");
        if (excluded_count) {
          INFO (str(excluded_count) + " of " + str(fixels.size()) + " fixels were tracked, but have been excluded from optimisation due to inadequate reconstruction;");
          INFO ("these contribute " + str (100.0 * excluded_cf_sum / cf) + "\% of the initial cost function");
        } else if (min_td_frac) {
          INFO ("No fixels were excluded from optimisation due to poor reconstruction");
        }
      }




      void TckFactor::test_streamline_length_scaling()
      {
        VAR (calc_cost_function());

        for (vector<Fixel>::iterator i = fixels.begin(); i != fixels.end(); ++i)
          i->clear_TD();

        coefficients.resize (num_tracks(), 0.0);
        TD_sum = 0.0;

        for (SIFT::track_t track_index = 0; track_index != num_tracks(); ++track_index) {
          const SIFT::TrackContribution& tck_cont (*contributions[track_index]);
          const double weight = 1.0 / tck_cont.get_total_length();
          coefficients[track_index] = std::log (weight);
          for (size_t i = 0; i != tck_cont.dim(); ++i)
            fixels[tck_cont[i].get_fixel_index()] += weight * tck_cont[i].get_length();
          TD_sum += weight * tck_cont.get_total_contribution();
        }

        VAR (calc_cost_function());

        // Also test varying mu; produce a scatter plot
        const double actual_TD_sum = TD_sum;
        std::ofstream out ("mu.csv", std::ios_base::trunc);
        for (int i = -1000; i != 1000; ++i) {
          const double factor = std::pow (10.0, double(i) / 1000.0);
          TD_sum = factor * actual_TD_sum;
          out << str(factor) << "," << str(calc_cost_function()) << "\n";
        }
        out << "\n";
        out.close();

        TD_sum = actual_TD_sum;
      }



      void TckFactor::calc_afcsa()
      {

        CONSOLE ("Cost function before linear optimisation is " + str(calc_cost_function()) + ")");

        try {
          coefficients = decltype(coefficients)::Zero (num_tracks());
        } catch (...) {
          throw Exception ("Error assigning memory for streamline weights vector");
        }

        class Functor
        { NOMEMALIGN
          public:
            Functor (TckFactor& master) :
                master (master),
                fixed_mu (master.mu()) { }
            Functor (const Functor&) = default;
            bool operator() (const SIFT::TrackIndexRange& range) const {
              for (SIFT::track_t track_index = range.first; track_index != range.second; ++track_index) {
                const SIFT::TrackContribution& tckcont = *master.contributions[track_index];
                double sum_afd = 0.0;
                for (size_t f = 0; f != tckcont.dim(); ++f) {
                  const size_t fixel_index = tckcont[f].get_fixel_index();
                  const Fixel& fixel = master.fixels[fixel_index];
                  const float length = tckcont[f].get_length();
                  sum_afd += fixel.get_weight() * fixel.get_FOD() * (length / fixel.get_orig_TD());
                }
                if (sum_afd && tckcont.get_total_contribution()) {
                  const double afcsa = sum_afd / tckcont.get_total_contribution();
                  master.coefficients[track_index] = std::max (master.min_coeff, std::log (afcsa / fixed_mu));
                } else {
                  master.coefficients[track_index] = master.min_coeff;
                }
              }
              return true;
            }
          private:
            TckFactor& master;
            const double fixed_mu;
        };
        {
          SIFT::TrackIndexRangeWriter writer (SIFT_TRACK_INDEX_BUFFER_SIZE, num_tracks());
          Functor functor (*this);
          Thread::run_queue (writer, SIFT::TrackIndexRange(), Thread::multi (functor));
        }

        for (vector<Fixel>::iterator i = fixels.begin(); i != fixels.end(); ++i) {
          i->clear_TD();
          i->clear_mean_coeff();
        }
        {
          SIFT::TrackIndexRangeWriter writer (SIFT_TRACK_INDEX_BUFFER_SIZE, num_tracks());
          FixelUpdater worker (*this);
          Thread::run_queue (writer, SIFT::TrackIndexRange(), Thread::multi (worker));
        }

        CONSOLE ("Cost function after linear optimisation is " + str(calc_cost_function()) + ")");

      }




      void TckFactor::estimate_factors()
      {

        try {
          coefficients = decltype(coefficients)::Zero (num_tracks());
        } catch (...) {
          throw Exception ("Error assigning memory for streamline weights vector");
        }

        const double init_cf = calc_cost_function();
        double cf_data = init_cf;
        double new_cf = init_cf;
        double prev_cf = init_cf;
        double cf_reg = 0.0;
        const double required_cf_change = -min_cf_decrease_percentage * init_cf;

        unsigned int nonzero_streamlines = 0;
        for (SIFT::track_t i = 0; i != num_tracks(); ++i) {
          if (contributions[i] && contributions[i]->dim())
            ++nonzero_streamlines;
        }

        unsigned int iter = 0;

        auto display_func = [&](){ return printf("    %5u        %3.3f%%         %2.3f%%        %u", iter, 100.0 * cf_data / init_cf, 100.0 * cf_reg / init_cf, nonzero_streamlines); };
        CONSOLE ("  Iteration     CF (data)      CF (reg)     Streamlines");
        ProgressBar progress ("");

        // Keep track of total exclusions, not just how many are removed in each iteration
        size_t total_excluded = 0;
        for (size_t i = 1; i != fixels.size(); ++i) {
          if (fixels[i].is_excluded())
            ++total_excluded;
        }

        std::unique_ptr<std::ofstream> csv_out;
        if (!csv_path.empty()) {
          csv_out.reset (new std::ofstream());
          csv_out->open (csv_path.c_str(), std::ios_base::trunc);
          (*csv_out) << "Iteration,Cost_data,Cost_reg_tik,Cost_reg_tv,Cost_reg,Cost_total,Streamlines,Fixels_excluded,Step_min,Step_mean,Step_mean_abs,Step_var,Step_max,Coeff_min,Coeff_mean,Coeff_mean_abs,Coeff_var,Coeff_max,Coeff_norm,\n";
          (*csv_out) << "0," << init_cf << ",0,0,0," << init_cf << "," << nonzero_streamlines << "," << total_excluded << ",0,0,0,0,0,0,0,0,0,0,0,\n";
          csv_out->flush();
        }

        // Initial estimates of how each weighting coefficient is going to change
        // The ProjectionCalculator classes overwrite these in place, so do an initial allocation but
        //   don't bother wiping it at every iteration
        //vector<float> projected_steps (num_tracks(), 0.0);

        // Logging which fixels need to be excluded from optimisation in subsequent iterations,
        //   due to driving streamlines to unwanted high weights
        BitSet fixels_to_exclude (fixels.size());

        do {

          ++iter;
          prev_cf = new_cf;

          // Line search to optimise each coefficient
          StreamlineStats step_stats, coefficient_stats;
          nonzero_streamlines = 0;
          fixels_to_exclude.clear();
          double sum_costs = 0.0;
          {
            SIFT::TrackIndexRangeWriter writer (SIFT_TRACK_INDEX_BUFFER_SIZE, num_tracks());
            //CoefficientOptimiserGSS worker (*this, /*projected_steps,*/ step_stats, coefficient_stats, nonzero_streamlines, fixels_to_exclude, sum_costs);
            //CoefficientOptimiserQLS worker (*this, /*projected_steps,*/ step_stats, coefficient_stats, nonzero_streamlines, fixels_to_exclude, sum_costs);
            CoefficientOptimiserIterative worker (*this, /*projected_steps,*/ step_stats, coefficient_stats, nonzero_streamlines, fixels_to_exclude, sum_costs);
            Thread::run_queue (writer, SIFT::TrackIndexRange(), Thread::multi (worker));
          }
          step_stats.normalise();
          coefficient_stats.normalise();
          indicate_progress();

          // Perform fixel exclusion
          const size_t excluded_count = fixels_to_exclude.count();
          if (excluded_count) {
            DEBUG (str(excluded_count) + " fixels excluded this iteration");
            for (size_t f = 0; f != fixels.size(); ++f) {
              if (fixels_to_exclude[f])
                fixels[f].exclude();
            }
            total_excluded += excluded_count;
          }

          // Multi-threaded calculation of updated streamline density, and mean weighting coefficient, in each fixel
          for (vector<Fixel>::iterator i = fixels.begin(); i != fixels.end(); ++i) {
            i->clear_TD();
            i->clear_mean_coeff();
          }
          {
            SIFT::TrackIndexRangeWriter writer (SIFT_TRACK_INDEX_BUFFER_SIZE, num_tracks());
            FixelUpdater worker (*this);
            Thread::run_queue (writer, SIFT::TrackIndexRange(), Thread::multi (worker));
          }
          // Scale the fixel mean coefficient terms (each streamline in the fixel is weighted by its length)
          for (vector<Fixel>::iterator i = fixels.begin(); i != fixels.end(); ++i)
            i->normalise_mean_coeff();
          indicate_progress();

          cf_data = calc_cost_function();

          // Calculate the cost of regularisation, given the updates to both the
          //   streamline weighting coefficients and the new fixel mean coefficients
          // Log different regularisation costs separately
          double cf_reg_tik = 0.0, cf_reg_tv = 0.0;
          {
            SIFT::TrackIndexRangeWriter writer (SIFT_TRACK_INDEX_BUFFER_SIZE, num_tracks());
            RegularisationCalculator worker (*this, cf_reg_tik, cf_reg_tv);
            Thread::run_queue (writer, SIFT::TrackIndexRange(), Thread::multi (worker));
          }
          cf_reg_tik *= reg_multiplier_tikhonov;
          cf_reg_tv  *= reg_multiplier_tv;

          double cf_reg_micro = 0.0;
          if (has_microstructure) {
            for (SIFT::track_t i = 0; i != num_tracks(); ++i) {
              if (micro_blend[i] > 0.0)
                cf_reg_micro += reg_multiplier_micro * micro_blend[i] * Math::pow2 (coefficients[i] - std::log (microstructure_af[i]));
            }
          }

          cf_reg = cf_reg_tik + cf_reg_tv + cf_reg_micro;

          new_cf = cf_data + cf_reg;

          if (!csv_path.empty()) {
            (*csv_out) << str (iter) << "," << str (cf_data) << "," << str (cf_reg_tik) << "," << str (cf_reg_tv) << "," << str (cf_reg) << "," << str (new_cf) << "," << str (nonzero_streamlines) << "," << str (total_excluded) << ","
                << str (step_stats       .get_min()) << "," << str (step_stats       .get_mean()) << "," << str (step_stats       .get_mean_abs()) << "," << str (step_stats       .get_var()) << "," << str (step_stats       .get_max()) << ","
                << str (coefficient_stats.get_min()) << "," << str (coefficient_stats.get_mean()) << "," << str (coefficient_stats.get_mean_abs()) << "," << str (coefficient_stats.get_var()) << "," << str (coefficient_stats.get_max()) << ","
                << str (coefficient_stats.get_var() * (num_tracks() - 1))
                << ",\n";
            csv_out->flush();
          }

          progress.update (display_func);

          // Leaving out testing the fixel exclusion mask criterion; doesn't converge, and results in CF increase
        } while (((new_cf - prev_cf < required_cf_change) || (iter < min_iters) /* || !fixels_to_exclude.empty() */ ) && (iter < max_iters));
      }




      void TckFactor::output_factors (const std::string& path) const
      {
        if (size_t(coefficients.size()) != contributions.size())
          throw Exception ("Cannot output weighting factors if they have not first been estimated!");
        decltype(coefficients) weights;
        try {
          weights.resize (coefficients.size());
        } catch (...) {
          WARN ("Unable to assign memory for output factor file: \"" + Path::basename(path) + "\" not created");
          return;
        }
        for (SIFT::track_t i = 0; i != num_tracks(); ++i)
          weights[i] = (coefficients[i] == min_coeff || !std::isfinite(coefficients[i])) ?
                        0.0 :
                        std::exp (coefficients[i]);
        save_vector (weights, path);
      }



      void TckFactor::output_coefficients (const std::string& path) const
      {
        save_vector (coefficients, path);
      }




      void TckFactor::output_all_debug_images (const std::string& prefix) const
      {

        Model<Fixel>::output_all_debug_images (prefix);

        if (!coefficients.size())
          return;

        vector<double> mins   (fixels.size(), 100.0);
        vector<double> stdevs (fixels.size(), 0.0);
        vector<double> maxs   (fixels.size(), -100.0);
        vector<size_t> zeroed (fixels.size(), 0);

        {
          ProgressBar progress ("Generating streamline coefficient statistic images", num_tracks());
          for (SIFT::track_t i = 0; i != num_tracks(); ++i) {
            const double coeff = coefficients[i];
            const SIFT::TrackContribution& this_contribution (*contributions[i]);
            if (coeff > min_coeff) {
              for (size_t j = 0; j != this_contribution.dim(); ++j) {
                const size_t fixel_index = this_contribution[j].get_fixel_index();
                const double mean_coeff = fixels[fixel_index].get_mean_coeff();
                mins  [fixel_index] = std::min (mins[fixel_index], coeff);
                stdevs[fixel_index] += Math::pow2 (coeff - mean_coeff);
                maxs  [fixel_index] = std::max (maxs[fixel_index], coeff);
              }
            } else {
              for (size_t j = 0; j != this_contribution.dim(); ++j)
                ++zeroed[this_contribution[j].get_fixel_index()];
            }
            ++progress;
          }
        }

        for (size_t i = 1; i != fixels.size(); ++i) {
          if (mins[i] == 100.0)
            mins[i] = 0.0;
          stdevs[i] = (fixels[i].get_count() > 1) ? (std::sqrt (stdevs[i] / float(fixels[i].get_count() - 1))) : 0.0;
          if (maxs[i] == -100.0)
            maxs[i] = 0.0;
        }

        using MR::Fixel::Legacy::FixelMetric;
        Header H_fixel (Fixel_map<Fixel>::header());
        H_fixel.datatype() = DataType::UInt64;
        H_fixel.datatype().set_byte_order_native();
        H_fixel.keyval()[MR::Fixel::Legacy::name_key] = str(typeid(FixelMetric).name());
        H_fixel.keyval()[MR::Fixel::Legacy::size_key] = str(sizeof(FixelMetric));

        MR::Fixel::Legacy::Image<FixelMetric> count_image    (prefix + "_count.msf",        H_fixel);
        MR::Fixel::Legacy::Image<FixelMetric> min_image      (prefix + "_coeff_min.msf",    H_fixel);
        MR::Fixel::Legacy::Image<FixelMetric> mean_image     (prefix + "_coeff_mean.msf",   H_fixel);
        MR::Fixel::Legacy::Image<FixelMetric> stdev_image    (prefix + "_coeff_stdev.msf",  H_fixel);
        MR::Fixel::Legacy::Image<FixelMetric> max_image      (prefix + "_coeff_max.msf",    H_fixel);
        MR::Fixel::Legacy::Image<FixelMetric> zeroed_image   (prefix + "_coeff_zeroed.msf", H_fixel);
        MR::Fixel::Legacy::Image<FixelMetric> excluded_image (prefix + "_excluded.msf",     H_fixel);

        VoxelAccessor v (accessor());
        for (auto l = Loop(v) (v, count_image, min_image, mean_image, stdev_image, max_image, zeroed_image, excluded_image); l; ++l) {
          if (v.value()) {

            count_image   .value().set_size ((*v.value()).num_fixels());
            min_image     .value().set_size ((*v.value()).num_fixels());
            mean_image    .value().set_size ((*v.value()).num_fixels());
            stdev_image   .value().set_size ((*v.value()).num_fixels());
            max_image     .value().set_size ((*v.value()).num_fixels());
            zeroed_image  .value().set_size ((*v.value()).num_fixels());
            excluded_image.value().set_size ((*v.value()).num_fixels());

            size_t index = 0;
            for (typename Fixel_map<Fixel>::ConstIterator iter = begin (v); iter; ++iter, ++index) {
              const size_t fixel_index = size_t(iter);
              FixelMetric fixel_metric (iter().get_dir().cast<float>(), iter().get_FOD(), iter().get_count());
              count_image   .value()[index] = fixel_metric;
              fixel_metric.value = mins[fixel_index];
              min_image     .value()[index] = fixel_metric;
              fixel_metric.value = iter().get_mean_coeff();
              mean_image    .value()[index] = fixel_metric;
              fixel_metric.value = stdevs[fixel_index];
              stdev_image   .value()[index] = fixel_metric;
              fixel_metric.value = maxs[fixel_index];
              max_image     .value()[index] = fixel_metric;
              fixel_metric.value = zeroed[fixel_index];
              zeroed_image  .value()[index] = fixel_metric;
              fixel_metric.value = iter().is_excluded() ? 0.0 : 1.0;
              excluded_image.value()[index] = fixel_metric;
            }

          }
        }
      }






      }
    }
  }
}



