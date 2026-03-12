#include <gtest/gtest.h>
#include "io/value_parser.hpp"

using namespace mpfem;

class ValueParserTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ValueParserTest, ParseSimpleNumber) {
    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseFirstNumber("3.14", value));
    EXPECT_DOUBLE_EQ(value, 3.14);
}

TEST_F(ValueParserTest, ParseNumberWithUnit) {
    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseFirstNumber("20[mV]", value));
    EXPECT_DOUBLE_EQ(value, 20.0);
}

TEST_F(ValueParserTest, ParseNumberWithBracket) {
    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseFirstNumber("9[cm]", value));
    EXPECT_DOUBLE_EQ(value, 9.0);
}

TEST_F(ValueParserTest, ParseScientificNotation) {
    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseFirstNumber("1.72e-8[ohm*m]", value));
    EXPECT_DOUBLE_EQ(value, 1.72e-8);
}

TEST_F(ValueParserTest, ParseNegativeNumber) {
    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseFirstNumber("-5.5", value));
    EXPECT_DOUBLE_EQ(value, -5.5);
}

TEST_F(ValueParserTest, ParseInteger) {
    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseFirstNumber("42", value));
    EXPECT_DOUBLE_EQ(value, 42.0);
}

TEST_F(ValueParserTest, ParseEmptyString) {
    double value = 0.0;
    EXPECT_FALSE(ValueParser::parseFirstNumber("", value));
}

TEST_F(ValueParserTest, ParseNoNumber) {
    double value = 0.0;
    EXPECT_FALSE(ValueParser::parseFirstNumber("[V]", value));
}

TEST_F(ValueParserTest, ParseWithVariables) {
    std::map<std::string, double> variables;
    variables["Vtot"] = 0.02;
    variables["htc"] = 5.0;

    double value = 0.0;
    EXPECT_TRUE(ValueParser::parseWithVariables("Vtot", variables, value));
    EXPECT_DOUBLE_EQ(value, 0.02);

    EXPECT_TRUE(ValueParser::parseWithVariables("htc", variables, value));
    EXPECT_DOUBLE_EQ(value, 5.0);
}

TEST_F(ValueParserTest, ParseNumberWithVariablesFallback) {
    std::map<std::string, double> variables;
    variables["Vtot"] = 0.02;

    double value = 0.0;
    // Direct number parsing should still work
    EXPECT_TRUE(ValueParser::parseWithVariables("0.02[V]", variables, value));
    EXPECT_DOUBLE_EQ(value, 0.02);
}

TEST_F(ValueParserTest, EvalFunction) {
    std::map<std::string, double> variables;
    variables["L"] = 0.09;
    variables["htc"] = 5.0;

    EXPECT_DOUBLE_EQ(ValueParser::eval("L", variables), 0.09);
    EXPECT_DOUBLE_EQ(ValueParser::eval("htc", variables), 5.0);
    EXPECT_DOUBLE_EQ(ValueParser::eval("10.5", variables), 10.5);
}

TEST_F(ValueParserTest, EvalMissingVariable) {
    std::map<std::string, double> variables;
    
    // Missing variable should return 0.0
    EXPECT_DOUBLE_EQ(ValueParser::eval("unknown", variables), 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
