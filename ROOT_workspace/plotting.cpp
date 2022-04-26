//**************************************************************
// H2tautau analysis. Plotting code: MC sim. and Real Data
// Author: Domenico Riccardi
// Creation Date: 10/04/2022
// Last Update: 24/04/2022
//***************************************************************

// Include libraries
#include <iostream>
#include <map>
#include <vector>
#include <utility>
#define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#include <experimental/filesystem>

#include "ROOT/RDataFrame.hxx"
#include "TCanvas.h"
#include "TLegend.h"
#include "THStack.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TColor.h"

// namespace-alias-definition: makes name "RFilter" a synonym for another namespace
using RFilter = ROOT::RDF::RInterface<ROOT::Detail::RDF::RJittedFilter>;

/*
 * Plotting function.
 * This function template produces the plot of the variable's distribution passed as argument.
 */
template <typename T>
void plotting(T samples, std::string variable, const char *title, std::string xlabel_, std::string unit, int bins, float lim_left, float lim_right) {
    std::cout << "\tPlotting: " << variable << std::endl;
    
    TCanvas *c = new TCanvas("c","c", 1000, 700);
    gStyle->SetOptStat(0);
    gStyle->SetTitleFontSize(0.05);
    // Define six 1D histograms for each channel
    auto h1 = samples.at("GluGluToHToTauTau_selected").Histo1D({"", "", bins, lim_left, lim_right}, variable, "weight");
    auto h2 = samples.at("VBF_HToTauTau_selected").Histo1D({"", "", bins, lim_left, lim_right}, variable, "weight");
    auto h3 = samples.at("DYJetsToLL_selected").Histo1D({"", "", bins, lim_left, lim_right}, variable, "weight");
    auto h4 = samples.at("TTbar_selected").Histo1D({"", "", bins, lim_left, lim_right}, variable, "weight");
    auto h5 = samples.at("W*JetsToLNu_selected").Histo1D({"", "", bins, lim_left, lim_right}, variable, "weight");
    auto h6 = samples.at("Run2012*_TauPlusX_selected").Histo1D({"", "", bins, lim_left, lim_right}, variable);
    
    std::vector<decltype(h1)> hist = {h1, h2, h3, h4, h5};
    std::vector<decltype(kRed+1)> color = {kRed+1, kMagenta-9, kGreen-3, kAzure+7, kOrange-3};
    // Assign the colors of each histogram
    for (int i=0; i<hist.size(); i++) {
        hist[i]->SetLineColor(color[i]);
        hist[i]->SetFillColor(color[i]);
    }
    // Create a stacked histrogram with MC simulation samples
    THStack *hs = new THStack("hs", "");
    for (auto &h : hist) {
        hs->Add(h.GetPtr());
    }
    hs->Draw("HISTO");
    // Draw the Data with black points and error bars
    h6->SetMarkerStyle(kFullCircle);
    h6->SetMarkerColor(kBlack);
    h6->SetLineColor(kBlack);
    h6->DrawClone("ESAME");
    
    // Set the x-y axis labels and calculate the binning
    auto binning = (lim_right - lim_left)/bins;
    std::stringstream ss;
    ss.precision(2);
    ss << binning;
    std::string y_label = "Candidates / " + ss.str() + " " + unit;
    std::string x_label = xlabel_ + " " + unit;
    hs->GetXaxis()->SetTitle(x_label.c_str());
    hs->GetYaxis()->SetTitle(y_label.c_str());
    hs->SetTitle(title);
    
    // Draw a Tlegend for Real Data and MC samples
    TLegend *legend = new TLegend(0.65, 0.65, 0.98, 0.88);
    legend->AddEntry(h1.GetPtr(),"gg#rightarrow H #rightarrow#tau_{#mu}#tau_{h}","f");
    legend->AddEntry(h2.GetPtr(),"qq#rightarrow H #rightarrow#tau_{#mu}#tau_{h}","f");
    legend->AddEntry(h3.GetPtr(),"Z#rightarrow ll","f");
    legend->AddEntry(h4.GetPtr(),"t#bar{t}", "f");
    legend->AddEntry(h5.GetPtr(),"W+jets", "f");
    legend->AddEntry(h6.GetPtr(), "Data");
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->Draw();
    
    float xposition_lt = hs->GetXaxis()->GetXmax() *0.03 + hs->GetXaxis()->GetXmin();
    TLatex *lt = new TLatex(xposition_lt, hs->GetMaximum()*0.95,"#scale[0.8]{#splitline{CMS #bf{#it{Open Data}}}{#bf{11.5 fb^{-1} (8 TeV)}}}");
    lt->Draw("Same");
    
    // Save the plot in the correct directory
    std::string name = "./Distribution_plots/" + variable + ".pdf";
    c->SaveAs(name.c_str());
    c->Update();
}

int main()
{
    // Create a new folder to contain the distribution plots produced
    std::string dirName = "Distribution_plots";
    std::stringstream bufH;
    bufH << dirName;
    if (!std::experimental::filesystem::exists(bufH.str()))
    {
        std::experimental::filesystem::create_directories(bufH.str());
    }
    
    // Define a std::map that contains the pair sample-RDataFrame and a std::vector of ROOT files' names
    std::map<std::string, RFilter> samples;
    std::vector<std::string> ROOT_files = {
        "GluGluToHToTauTau_selected",
        "VBF_HToTauTau_selected",
        "DYJetsToLL_selected",
        "TTbar_selected",
        "W*JetsToLNu_selected",
        "Run2012*_TauPlusX_selected"
    };
    
    // For loop to process the events in ROOT files and to apply final cuts to reduce the background (mostly W+jets)
    for (const std::string &channel: ROOT_files) {
        std::cout << "\tProcessing: " << channel << std::endl;
        ROOT::RDataFrame d("Events", channel + ".root");
        auto d_filter = d.Filter("mt_mu<30 & muon_iso<0.1")
                         .Filter("tau_pt>25 & muon_pt>20");
        samples.insert(std::pair<std::string,RFilter>(channel, d_filter));
        std::cout << "\tNumber of events: " << *samples.at(channel).Count() << std::endl;
    }
    
    // Plot distribution of physical variables by using the plotting function (defined above)
    std::cout << "*************************** Plotting distribution ********************************" << std::endl;
    plotting(samples, "m_vis", "Visible mass m_{#mu}+m_{#tau}", "m_{vis}", "[GeV]", 30, 20, 140);
    plotting(samples, "pt_vis", "Visible transverse momentum", "p_{T,vis}", "[GeV]", 30, 0, 60);
    plotting(samples, "tau_pt", "#tau transverse momentum", "p_{T}(#tau )", "[GeV]", 30, 17, 70);
    plotting(samples, "muon_pt", "#mu transverse momentum", "p_{T}(#mu)", "[GeV]", 30, 17, 70);
    plotting(samples, "muon_eta", "#mu pseudorapidity", "#eta(#mu)", "", 30, -2.1, 2.1);
    plotting(samples, "tau_eta", "#tau pseudorapidity", "#eta(#tau)", "", 30, -2.3, 2.3);
    plotting(samples, "muon_phi", "#mu angular distribution", "#phi(#mu)", "", 30, -3.14, 3.14);
    plotting(samples, "tau_phi", "#tau angular distribution", "#phi(#tau)", "", 30, -3.14, 3.14);
    plotting(samples, "muon_iso", "#mu isolation", "#mu ISO", "", 30, 0, 0.1);
    plotting(samples, "tau_iso", "#tau isolation", "#tau ISO", "", 30, 0, 0.1);
    plotting(samples, "muon_charge", "#mu charge", "q(#mu)", "", 2, -2, 2);
    plotting(samples, "tau_charge", "#tau charge", "q(#tau)", "", 2, -2, 2);
    plotting(samples, "MET_pt", "Missing transverse energy p_{T}", "p_{T}(MET)", "[GeV]", 30, 0, 60);
    plotting(samples, "MET_phi", "Missing transverse energy #phi", "#phi(MET)", "", 30, -3.14, 3.14);
    plotting(samples, "muon_m", "#mu mass", "m(#mu)", "[GeV]", 30, 0, 0.2);
    plotting(samples, "tau_m", "#tau mass", "m(#mu)", "[GeV]", 30, 0, 2);
    plotting(samples, "mt_mu", "#mu transverse mass", "m_{T}(#mu)", "[GeV]", 30, 0, 100);
    plotting(samples, "mt_tau", "#tau transverse mass", "m_{T}(#tau)", "[GeV]", 30, 0, 100);
    plotting(samples, "mt_mu", "#mu transverse mass", "m_{T}(#mu)", "[GeV]", 30, 0, 100);
    plotting(samples, "jpt_1", "Leading jet transverse momentum", "p_{T}(j^{leading})", "[GeV]", 30, 30, 70);
    plotting(samples, "jpt_2", "Trailing jet transverse momentum", "p_{T}(j^{trailing})", "[GeV]", 30, 30, 70);
    plotting(samples, "jeta_1", "Leading jet pseudorapidity", "#eta(j^{leading})", "", 30, -4.7, 4.7);
    plotting(samples, "jeta_2", "Trailing jet pseudorapidity", "#eta(j^{trailing})", "", 30, -4.7, 4.7);
    plotting(samples, "jphi_1", "Leading jet angular distribution", "#phi(j^{leading})", "", 30, -3.14, 3.14);
    plotting(samples, "jphi_2", "Trailing jet angular distribution", "#phi(j^{trailing})", "", 30, -3.14, 3.14);
    plotting(samples, "jm_1", "Leading jet mass", "m(j^{leading})", "[GeV]", 30, 0, 20);
    plotting(samples, "jm_2", "Trailing jet mass", "m(j^{trailing})", "[GeV]", 30, 0, 20);
    plotting(samples, "jbtag_1", "Leading jet btag", "b-tag(j^{leading})", "", 30, 0, 1);
    plotting(samples, "jbtag_2", "Trailing jet btag", "b-tag(j^{trailing})", "", 30, 0, 1);
    plotting(samples, "PV_npvs", "Number of primary vertex", "NPV", "", 30, 0, 30);
    plotting(samples, "nGoodJets", "Number of good jets", "Num. jets", "", 5, 0, 5);
    plotting(samples, "jj_m", "Di-jet mass distribution", "m(jj)", "[GeV]", 30, 0, 400);
    plotting(samples, "jj_pt", "Di-jet pt distribution", "p_{T}(jj)", "[GeV]", 30, 0, 200);
    plotting(samples, "jj_delta", "Di-jet #Delta #eta", "#Delta #eta(jj)", "", 30, -9.4, 9.4);
    
}
